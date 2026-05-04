"""Benchmark RAG answer quality on the user's D&D 2014 5e workspace.

Targets the failure mode the user reported: open-ended exploratory
queries like "list unique sounding NPCs from the books" come back with
the model answering a different question. The benchmark probes:

  - Whether the baseline cfg actually retrieves relevant chunks
  - Whether HyDE / multi-query / agentic helps for exploratory queries
  - Whether a stricter system prompt prevents the LLM from drifting
  - Whether rule-lookup questions (where baseline should work) regress
    when we turn the harder strategies on

Output: a Markdown report + JSON dump under bench/reports/.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag.embed import make_embedder
from ez_rag.generate import (
    SYSTEM_PROMPT_RAG, _ollama_chat, _llm_complete, answer, detect_backend,
    generate_query_variations,
)
from ez_rag.index import Index
from ez_rag.retrieve import smart_retrieve, agentic_retrieve
from ez_rag.workspace import Workspace


# ============================================================================
# Test questions — biased toward the user's reported failure mode
# ============================================================================

@dataclass
class Question:
    text: str
    category: str           # "exploratory" | "rule" | "comparison" | "multi-step"
    expected_keywords: list[str] = field(default_factory=list)
    notes: str = ""


QUESTIONS: list[Question] = [
    # ---------- Exploratory (the user's pain point) ----------
    Question("List some unique-sounding NPCs from the books with their names.",
             "exploratory",
             ["NPC", "name"],
             "USER-REPORTED FAILURE CASE"),
    Question("What are some interesting magical items a DM could give a level 5 party?",
             "exploratory", ["+1", "rare", "uncommon", "wondrous"]),
    Question("Suggest some non-violent ways to resolve a goblin encounter.",
             "exploratory", ["persuasion", "deception", "intimidation"]),
    Question("List notable villains and antagonists from the published adventures.",
             "exploratory", ["villain", "antagonist"]),
    Question("What are some classic D&D dungeons or locations a new DM could borrow?",
             "exploratory", ["dungeon", "tomb", "castle"]),
    Question("Give examples of memorable taverns or inns mentioned in any of the books.",
             "exploratory", ["tavern", "inn"]),
    Question("Suggest some unique character backgrounds beyond the standard ones.",
             "exploratory", ["background"]),
    Question("List interesting deities or gods that PCs could worship.",
             "exploratory", ["god", "deity", "domain"]),
    Question("What are some fun magical weapons that are not just '+1 sword'?",
             "exploratory", ["magic weapon", "rare"]),
    Question("Name some intelligent monsters that could be diplomatic encounters.",
             "exploratory", ["monster", "intelligence"]),
    Question("List some short adventure hooks or quest seeds.",
             "exploratory", ["adventure", "quest", "hook"]),
    Question("What are the most iconic dragons in D&D 5e lore?",
             "exploratory", ["dragon", "ancient"]),
    Question("Suggest some interesting curses or hexes for a campaign.",
             "exploratory", ["curse", "hex"]),
    Question("Give examples of legendary artifacts in 5e.",
             "exploratory", ["artifact", "legendary"]),
    Question("List unusual or weird monsters that players might not have seen before.",
             "exploratory", ["monster"]),

    # ---------- Rule lookup (baseline should work; checks for regressions) ----------
    Question("How does the grappling rule work in 5e?",
             "rule", ["athletics", "strength", "contested"]),
    Question("What is the AC of plate armor?",
             "rule", ["18", "armor class"]),
    Question("How many spells can a level 3 wizard prepare?",
             "rule", ["intelligence", "modifier", "level"]),
    Question("What does the prone condition do?",
             "rule", ["disadvantage", "advantage", "movement"]),
    Question("How does concentration work for spells?",
             "rule", ["concentration", "constitution"]),
    Question("What are the rules for opportunity attacks?",
             "rule", ["opportunity", "reaction"]),
    Question("How does the surprise rule work in combat?",
             "rule", ["surprise", "initiative"]),
    Question("What does cover do mechanically?",
             "rule", ["half cover", "three-quarters", "+2", "+5"]),
    Question("How do death saving throws work?",
             "rule", ["death save", "DC 10", "stable"]),
    Question("What is the difference between a ritual spell and a normal spell?",
             "rule", ["ritual", "10 minutes"]),
    Question("How does the dodge action work?",
             "rule", ["dodge", "disadvantage"]),
    Question("What is passive perception and how is it calculated?",
             "rule", ["passive perception", "10", "wisdom"]),
    Question("How does the help action work in combat?",
             "rule", ["help", "advantage"]),
    Question("What does the exhaustion condition do?",
             "rule", ["exhaustion", "level"]),
    Question("How long does a short rest take?",
             "rule", ["short rest", "1 hour"]),

    # ---------- Comparison ----------
    Question("Compare the Fighter and Barbarian at level 5 — strengths and weaknesses.",
             "comparison", ["Fighter", "Barbarian"]),
    Question("How does the Wizard's spellcasting differ from the Sorcerer's?",
             "comparison", ["Wizard", "Sorcerer", "metamagic"]),
    Question("What's the difference between a Cleric and a Paladin?",
             "comparison", ["Cleric", "Paladin"]),
    Question("Compare the Rogue and the Monk for unarmed combat.",
             "comparison", ["Rogue", "Monk"]),
    Question("Druid versus Ranger — when would you pick each?",
             "comparison", ["Druid", "Ranger"]),
    Question("Compare half-elf and half-orc in combat roles.",
             "comparison", ["half-elf", "half-orc"]),
    Question("What's the difference between divine and arcane magic?",
             "comparison", ["divine", "arcane"]),
    Question("Compare the Warlock's pact options.",
             "comparison", ["Warlock", "pact"]),
    Question("How do the rules for grappling differ from shoving?",
             "comparison", ["grapple", "shove"]),
    Question("Sneak Attack vs Smite — how do they compare in damage output?",
             "comparison", ["sneak attack", "smite"]),

    # ---------- Multi-step / generative ----------
    Question("If I have a level 5 Battle Master Fighter, what feats should I consider next?",
             "multi-step", ["feat", "Battle Master"]),
    Question("Build a level 1 Variant Human character optimized for a Wild Magic Sorcerer.",
             "multi-step", ["Wild Magic", "Sorcerer", "Variant Human"]),
    Question("How would I run a stealth-heavy infiltration mission for a party of 4?",
             "multi-step", ["stealth", "infiltration"]),
    Question("Design a small encounter for 4 level 3 PCs that uses environmental hazards.",
             "multi-step", ["encounter", "hazard"]),
    Question("Walk me through resolving a contested grapple between a Fighter and a giant spider.",
             "multi-step", ["grapple", "spider"]),
    Question("How should I balance a boss fight for 5 level 8 characters?",
             "multi-step", ["challenge rating", "encounter"]),
    Question("What are the steps to cast Fireball at 4th level?",
             "multi-step", ["fireball", "upcast", "4th level"]),
    Question("If a party loses a member mid-dungeon, how do I scale the rest of the encounters?",
             "multi-step", ["challenge rating", "scale"]),
    Question("Give me a checklist for prepping a 3-hour D&D session.",
             "multi-step", ["session", "prep"]),
    Question("How do I introduce a new player mid-campaign?",
             "multi-step", ["new player"]),
    Question("Outline a simple downtime activity system for between adventures.",
             "multi-step", ["downtime"]),
    Question("Walk me through running a chase scene.",
             "multi-step", ["chase"]),
    Question("What's a good way to handle PCs splitting up?",
             "multi-step", ["split", "party"]),
    Question("How do I telegraph a deadly trap so players have a fair chance?",
             "multi-step", ["trap", "perception"]),
    Question("Plan a one-shot adventure set in a haunted manor.",
             "multi-step", ["haunt", "manor"]),
]


# ============================================================================
# Strategies
# ============================================================================

@dataclass
class Strategy:
    name: str
    description: str
    cfg_overrides: dict
    system_prompt: str = SYSTEM_PROMPT_RAG


# A stricter system prompt that pushes the model NOT to drift to a
# different question when the retrieved context is sparse. The user's
# complaint was the model answering questions that weren't asked —
# this is the antidote.
STRICT_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a document "
    "collection. You will be shown the user's question and a set of "
    "context excerpts retrieved from the documents.\n\n"
    "RULES:\n"
    "1. Answer ONLY the literal question the user asked. Do not pivot "
    "to a related but different question.\n"
    "2. If the context does not directly address the question, say so "
    "explicitly: 'The retrieved excerpts don't directly address this. "
    "Here's what I found that's adjacent: …'. Then summarize what IS "
    "in the context — don't invent.\n"
    "3. For 'list X' / 'name some X' questions: extract specific named "
    "examples from the context. Do not give general advice or "
    "definitions instead of a list.\n"
    "4. Always include citations like [filename, page N] from the "
    "Context block.\n"
    "5. If you genuinely cannot answer from the context, say 'I don't "
    "have enough information in the indexed documents to answer this' "
    "rather than guessing."
)

# Tighter system prompt specifically for "list / name some / give "
# examples" queries. Forces the model to extract specific named items
# from messy context instead of pivoting into general lore.
LIST_EXTRACTION_PROMPT = (
    "You are an information extraction assistant. The user is asking "
    "for a LIST of specific named items (people, places, items, "
    "creatures, etc.) that appear in the retrieved context.\n\n"
    "Your job:\n"
    "1. SCAN every context excerpt for proper nouns / specific named "
    "items that match the user's request. Look for capitalized names, "
    "table entries, sidebar entries, stat-block headers, and any "
    "other specific examples.\n"
    "2. Output a BULLETED LIST. Each bullet:\n"
    "     • <specific name> — <one short sentence of context if "
    "available> [filename, page N]\n"
    "3. Do NOT explain general concepts. Do NOT pivot to 'here's how "
    "X works'. Do NOT define the term the user is asking about. The "
    "user wants the LIST, not the explanation.\n"
    "4. If you find specific names buried inside paragraphs of "
    "unrelated text, EXTRACT THEM. Do not summarize the surrounding "
    "paragraph instead.\n"
    "5. If you find fewer than 3 specific examples, say:\n"
    "     'Only N specific examples found in the indexed excerpts:'\n"
    "   and list what you have. Do NOT pad with generic examples or "
    "guesses outside the context.\n"
    "6. If the context truly contains zero specific examples, say "
    "'I did not find specific named examples for this in the indexed "
    "documents.' Do NOT make up names."
)


def _is_list_query(text: str) -> bool:
    """Heuristic: does this question want a LIST of specific items?"""
    t = text.lower().strip()
    triggers = (
        "list ", "name some", "give examples", "give me some",
        "give me examples", "what are some", "suggest some",
        "examples of", "names of", "some interesting",
        "some unique", "unique sounding", "memorable",
    )
    return any(trig in t for trig in triggers)


def generate_list_hyde(query: str, cfg: Config) -> str:
    """HyDE tuned for 'list X' queries — produces an entity-rich
    hypothetical passage rather than a summary answer.

    Embedding entity-rich text matches stat-block / table / sidebar
    chunks far better than embedding a summary, which tends to hit
    explanatory prose instead.
    """
    prompt = (
        "The user is searching a reference book for SPECIFIC NAMED "
        "ITEMS. Write 2-3 sentences of hypothetical text that might "
        "appear in such a book and would CONTAIN the named items the "
        "user wants. Use proper nouns, capitalized names, and "
        "domain-specific terminology. Pack as many distinct named "
        "examples into the text as possible. Do NOT explain or define "
        "anything. Just write the entity-rich passage.\n\n"
        f"Question: {query}\n\nPassage:"
    )
    out = _llm_complete(cfg, prompt, max_tokens=200)
    if not out.strip():
        return query
    return f"{query}\n{out.strip()}"


STRATEGIES: list[Strategy] = [
    # ===== Final A/B comparison =====
    # auto_list_mode is the new in-app default (True). Compare against
    # the OFF state to confirm the win on list queries doesn't regress
    # on rule lookups / comparisons / multi-step questions.
    Strategy(
        name="off-mode",
        description="auto_list_mode=False — pre-fix behavior",
        cfg_overrides={"auto_list_mode": False},
    ),
    Strategy(
        name="on-mode",
        description="auto_list_mode=True — new default; auto-routes "
                    "list queries through entity-rich HyDE + extraction prompt",
        cfg_overrides={"auto_list_mode": True},
    ),
    # ===== Chapter expansion sweep — Loop 2 =====
    # Test what chapter_max_chars does to grounded-ness. Hypothesis:
    # large expansions cause the LLM to summarize chapters instead of
    # extracting from them, which the judge sees as "ungrounded."
    Strategy(
        name="ch-off",
        description="expand_to_chapter=False — only the original chunks reach the LLM",
        cfg_overrides={"expand_to_chapter": False, "context_window": 0},
    ),
    Strategy(
        name="ch-2k",
        description="chapter_max_chars=2000 — tight focused expansion",
        cfg_overrides={"expand_to_chapter": True, "chapter_max_chars": 2000},
    ),
    Strategy(
        name="ch-4k",
        description="chapter_max_chars=4000 — moderate focused expansion",
        cfg_overrides={"expand_to_chapter": True, "chapter_max_chars": 4000},
    ),
    Strategy(
        name="ch-8k",
        description="chapter_max_chars=8000 — middle ground",
        cfg_overrides={"expand_to_chapter": True, "chapter_max_chars": 8000},
    ),
    Strategy(
        name="ch-neighbor",
        description="No chapter expand, but ±1 neighbor chunks",
        cfg_overrides={"expand_to_chapter": False, "context_window": 1},
    ),
    # ===== Loop 3 — grounding interventions on top of ch-2k =====
    Strategy(
        name="cite-required",
        description="ch-2k + cite-required system prompt",
        cfg_overrides={"expand_to_chapter": True, "chapter_max_chars": 2000},
        system_prompt=(
            "You are answering a question using ONLY the provided context "
            "excerpts. Strict rules:\n"
            "1. Every claim of fact must be followed by an inline citation "
            "in the form [N] referring to the numbered context items.\n"
            "2. If a fact is not stated in the context, you may NOT "
            "include it. Even if you 'know' it from training, leave it out.\n"
            "3. If the context is insufficient to answer, say exactly: "
            "'The provided excerpts don't contain enough to answer this. "
            "Here's what they do say: ...' and summarize what's actually "
            "in the excerpts.\n"
            "4. Do not invent page numbers, names, or rules.\n"
            "5. Be concise and direct."
        ),
    ),
    Strategy(
        name="extract-verbatim",
        description="ch-2k + EXTRACTOR-only prompt (no paraphrase, no general knowledge)",
        cfg_overrides={"expand_to_chapter": True, "chapter_max_chars": 2000},
        system_prompt=(
            "You are an EXTRACTOR, not a summarizer. The user has a "
            "question; you have context excerpts. Your job:\n"
            "1. Find the specific phrases, sentences, and named items in "
            "the context that answer the question.\n"
            "2. Quote them or paraphrase them tightly. Stay close to source wording.\n"
            "3. Cite each extraction with [N] referring to the numbered "
            "context items.\n"
            "4. Do NOT add general knowledge. Do NOT explain concepts that "
            "aren't in the context. Do NOT invent details.\n"
            "5. If the context doesn't contain the answer, say: "
            "'Not found in the indexed documents.' and stop."
        ),
    ),
    # ===== Loop 4 — lost-in-middle reorder + tighter top_k =====
    Strategy(
        name="ch-2k-reorder",
        description="ch-2k + lost-in-middle reorder (highest rank at start AND end)",
        cfg_overrides={"expand_to_chapter": True, "chapter_max_chars": 2000,
                        "reorder_for_attention": True},
    ),
    Strategy(
        name="ch-2k-no-reorder",
        description="ch-2k + reorder OFF (control)",
        cfg_overrides={"expand_to_chapter": True, "chapter_max_chars": 2000,
                        "reorder_for_attention": False},
    ),
    Strategy(
        name="ch-2k-top4",
        description="ch-2k + reorder + top_k=4 (tight selection per research)",
        cfg_overrides={"expand_to_chapter": True, "chapter_max_chars": 2000,
                        "reorder_for_attention": True, "top_k": 4},
    ),
    Strategy(
        name="ch-2k-top12",
        description="ch-2k + reorder + top_k=12",
        cfg_overrides={"expand_to_chapter": True, "chapter_max_chars": 2000,
                        "reorder_for_attention": True, "top_k": 12},
    ),
    # ===== Loop 5 — CRAG-style relevance filter =====
    Strategy(
        name="ch-2k-crag",
        description="ch-2k + CRAG chunk relevance filter (1 extra LLM call)",
        cfg_overrides={"expand_to_chapter": True, "chapter_max_chars": 2000,
                        "reorder_for_attention": False,
                        "crag_filter": True, "top_k": 12},
    ),
    Strategy(
        name="ch-2k-crag-top16",
        description="ch-2k + CRAG with wider initial pool (top_k=16) — "
                    "filter prunes back to relevant subset",
        cfg_overrides={"expand_to_chapter": True, "chapter_max_chars": 2000,
                        "reorder_for_attention": False,
                        "crag_filter": True, "top_k": 16},
    ),
    Strategy(
        name="cite-+-list",
        description="cite-required + auto-list mode for list-style queries",
        cfg_overrides={"expand_to_chapter": True, "chapter_max_chars": 2000,
                        "auto_list_mode": True},
        system_prompt=(
            "You are answering a question using ONLY the provided context "
            "excerpts.\n"
            "1. Every fact must have a [N] citation referring to numbered "
            "context items.\n"
            "2. If not in context, leave it out.\n"
            "3. For 'list X / name some X' questions, output a bulleted "
            "list of specific named items from the context with [N] "
            "citations. No definitions or general advice instead of names.\n"
            "4. If context is insufficient, say so explicitly.\n"
            "5. Be concise."
        ),
    ),
    # ===== Earlier exploration (kept for reference / reproducibility) =====
    Strategy(
        name="baseline",
        description="Current workspace cfg (hybrid + rerank + chapter expand)",
        cfg_overrides={"auto_list_mode": False},
    ),
    Strategy(
        name="strict-prompt",
        description="Same retrieval, stricter system prompt that "
                    "forbids drifting to a different question",
        cfg_overrides={},
        system_prompt=STRICT_SYSTEM_PROMPT,
    ),
    Strategy(
        name="multi-query",
        description="Fan out the query into 3 paraphrases, fuse with RRF",
        cfg_overrides={"multi_query": True, "top_k": 12},
        system_prompt=STRICT_SYSTEM_PROMPT,
    ),
    Strategy(
        name="hyde+strict",
        description="Generate hypothetical answer, embed that, plus strict prompt",
        cfg_overrides={"use_hyde": True, "top_k": 12},
        system_prompt=STRICT_SYSTEM_PROMPT,
    ),
    Strategy(
        name="agentic",
        description="LLM-driven iterative search (1 reflection cycle), strict prompt",
        cfg_overrides={"agentic": True, "agent_max_iterations": 2, "top_k": 8},
        system_prompt=STRICT_SYSTEM_PROMPT,
    ),
    Strategy(
        name="list-tuned",
        description="Auto-detect list-style queries; use entity-rich HyDE + "
                    "extraction-only system prompt. For non-list queries, "
                    "falls back to baseline retrieval + strict prompt.",
        cfg_overrides={"top_k": 12},
        system_prompt=LIST_EXTRACTION_PROMPT,   # used only for list queries
    ),
]


# ============================================================================
# Runner
# ============================================================================

@dataclass
class Result:
    question: str
    category: str
    strategy: str
    answer: str
    sources: list[str]
    seconds: float
    keywords_hit: int
    keywords_total: int
    error: str = ""


def make_cfg(workspace: Workspace, overrides: dict) -> Config:
    cfg = workspace.load_config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run_one(*, question: Question, strategy: Strategy,
            workspace: Workspace, embedder, index, save_cb=None) -> Result:
    cfg = make_cfg(workspace, strategy.cfg_overrides)
    t0 = time.perf_counter()
    try:
        # list-tuned strategy: auto-detect list-style queries and swap in
        # the entity-rich HyDE + extraction prompt. Non-list queries
        # fall back to plain strict-prompt behavior.
        is_list = (strategy.name == "list-tuned"
                    and _is_list_query(question.text))
        sys_prompt = strategy.system_prompt
        retrieval_query = question.text
        if is_list:
            retrieval_query = generate_list_hyde(question.text, cfg)
            sys_prompt = LIST_EXTRACTION_PROMPT
        elif strategy.name == "list-tuned":
            sys_prompt = STRICT_SYSTEM_PROMPT

        if strategy.cfg_overrides.get("agentic"):
            hits = agentic_retrieve(
                query=retrieval_query, embedder=embedder,
                index=index, cfg=cfg,
            )
        else:
            hits = smart_retrieve(
                query=retrieval_query, embedder=embedder,
                index=index, cfg=cfg,
            )
        # Build the LLM call ourselves so we can swap the system prompt.
        from ez_rag.generate import _build_user_prompt
        user_prompt = _build_user_prompt(question.text, hits)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = _ollama_chat(cfg, messages)
        sources = [
            f"{h.section or h.path}:{h.page or '?'}"
            for h in hits[:5]
        ]
    except Exception as ex:
        return Result(
            question=question.text, category=question.category,
            strategy=strategy.name, answer="", sources=[],
            seconds=time.perf_counter() - t0,
            keywords_hit=0,
            keywords_total=len(question.expected_keywords),
            error=f"{type(ex).__name__}: {ex}",
        )
    elapsed = time.perf_counter() - t0
    # Score: how many expected keywords appear in the answer.
    answer_lower = (text or "").lower()
    hits_kw = sum(1 for kw in question.expected_keywords
                   if kw.lower() in answer_lower)
    res = Result(
        question=question.text, category=question.category,
        strategy=strategy.name, answer=text or "(empty)",
        sources=sources, seconds=elapsed,
        keywords_hit=hits_kw,
        keywords_total=len(question.expected_keywords),
    )
    if save_cb:
        save_cb(res)
    return res


def _md_quote(s: str, indent: str = "> ") -> str:
    return "\n".join(indent + line for line in (s or "").splitlines())


def write_report(results: list[Result], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_question: dict[str, list[Result]] = {}
    for r in results:
        by_question.setdefault(r.question, []).append(r)

    # Aggregate per-strategy summary
    strat_totals: dict[str, dict[str, float]] = {}
    for r in results:
        s = strat_totals.setdefault(r.strategy, {
            "n": 0, "kw_hit": 0, "kw_total": 0, "seconds": 0.0,
            "errors": 0,
        })
        s["n"] += 1
        s["kw_hit"] += r.keywords_hit
        s["kw_total"] += r.keywords_total
        s["seconds"] += r.seconds
        if r.error:
            s["errors"] += 1

    lines = ["# D&D 5e RAG Quality Benchmark\n"]
    lines.append(f"Workspace: `C:\\Users\\jroun\\Desktop\\dnd books\\2014 (5e)`")
    lines.append(f"Questions: {len(by_question)}  ·  "
                  f"Strategies: {len(strat_totals)}  ·  "
                  f"Total runs: {len(results)}\n")

    lines.append("## Aggregate by strategy\n")
    lines.append("| Strategy | Keyword hit-rate | Avg time | Errors |")
    lines.append("|---|---|---|---|")
    for name, t in strat_totals.items():
        rate = t["kw_hit"] / max(1, t["kw_total"])
        avg_s = t["seconds"] / max(1, t["n"])
        lines.append(
            f"| {name} | {rate*100:.1f}% "
            f"({int(t['kw_hit'])}/{int(t['kw_total'])}) | "
            f"{avg_s:.1f}s | {int(t['errors'])} |"
        )
    lines.append("")

    # Per-question results, grouped by category
    cat_order = ["exploratory", "rule", "comparison", "multi-step"]
    for cat in cat_order:
        lines.append(f"## Category: {cat}\n")
        cat_questions = {q: rs for q, rs in by_question.items()
                          if rs and rs[0].category == cat}
        for q, rs in cat_questions.items():
            lines.append(f"### {q}")
            if rs[0].category == "exploratory" and "NPC" in q:
                lines.append("**[USER-REPORTED FAILURE CASE]**")
            for r in rs:
                lines.append(
                    f"\n**{r.strategy}** "
                    f"({r.seconds:.1f}s · "
                    f"{r.keywords_hit}/{r.keywords_total} kw)"
                )
                if r.error:
                    lines.append(f"_Error_: {r.error}")
                    continue
                lines.append(_md_quote(r.answer[:1200] +
                                        ("…" if len(r.answer) > 1200 else "")))
                if r.sources:
                    lines.append(f"\n_Sources_: {' · '.join(r.sources)}")
            lines.append("\n---\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=r"C:\Users\jroun\Desktop\dnd books\2014 (5e)")
    ap.add_argument("--out-dir", default=str(ROOT / "bench" / "reports"))
    ap.add_argument("--limit-questions", type=int, default=0,
                     help="Cap number of questions (0 = all)")
    ap.add_argument("--strategies", default="",
                     help="Comma-separated strategy names; default = all")
    args = ap.parse_args()

    ws = Workspace(Path(args.workspace))
    if not ws.is_initialized():
        print(f"workspace not initialized: {args.workspace}")
        return 1
    cfg = ws.load_config()
    if detect_backend(cfg) == "none":
        print("No LLM backend detected (Ollama not reachable).")
        return 1

    print(f"Workspace: {args.workspace}")
    print(f"LLM model: {cfg.llm_model}")
    print(f"Embedder:  {cfg.ollama_embed_model or cfg.embedder_model}")

    embedder = make_embedder(cfg)
    index = Index(ws.meta_db_path, embed_dim=embedder.dim)

    questions = QUESTIONS
    if args.limit_questions:
        questions = questions[:args.limit_questions]

    selected: list[Strategy] = STRATEGIES
    if args.strategies:
        wanted = {s.strip() for s in args.strategies.split(",") if s.strip()}
        selected = [s for s in STRATEGIES if s.name in wanted]
        if not selected:
            print(f"No matching strategies for: {wanted}")
            return 1

    print(f"Running {len(questions)} questions × "
          f"{len(selected)} strategies = "
          f"{len(questions) * len(selected)} total LLM calls\n")

    results: list[Result] = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"dnd5e-quality-{stamp}.json"
    md_path = out_dir / f"dnd5e-quality-{stamp}.md"

    def save_one(r: Result):
        results.append(r)
        # Periodic JSON dump so a crash doesn't lose all data.
        if len(results) % 10 == 0:
            json_path.write_text(
                json.dumps([asdict(x) for x in results], indent=2),
                encoding="utf-8",
            )

    for qi, q in enumerate(questions, start=1):
        print(f"[{qi}/{len(questions)}] {q.text[:70]}…")
        for s in selected:
            t0 = time.perf_counter()
            r = run_one(question=q, strategy=s, workspace=ws,
                          embedder=embedder, index=index, save_cb=save_one)
            print(f"   {s.name:14s} "
                  f"{r.seconds:5.1f}s  "
                  f"kw={r.keywords_hit}/{r.keywords_total}  "
                  f"{'ERR: ' + r.error if r.error else ''}")

    # Final dumps
    json_path.write_text(
        json.dumps([asdict(x) for x in results], indent=2),
        encoding="utf-8",
    )
    write_report(results, md_path)
    print(f"\n[OK] JSON:    {json_path}")
    print(f"[OK] Report:  {md_path}")

    # Quick stdout summary
    print("\n=== Summary ===")
    by_strat: dict[str, list[Result]] = {}
    for r in results:
        by_strat.setdefault(r.strategy, []).append(r)
    for name, rs in by_strat.items():
        kh = sum(r.keywords_hit for r in rs)
        kt = sum(r.keywords_total for r in rs)
        avg_s = sum(r.seconds for r in rs) / max(1, len(rs))
        errs = sum(1 for r in rs if r.error)
        print(f"  {name:14s}  kw {kh:3d}/{kt:3d} "
              f"({kh/max(1,kt)*100:5.1f}%)  "
              f"avg {avg_s:5.1f}s  errs {errs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

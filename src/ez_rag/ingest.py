"""Top-level ingest pipeline: walk docs/, parse, chunk, embed, upsert."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chapters import detect_chapters
from .chunker import chunk_sections
from .config import Config
from .embed import make_embedder
from .generate import (
    contextualize_chunk, correct_garbled_text, detect_backend,
    inspect_text_quality,
)
from .index import Index, file_sha256
from .models import unload_ollama_model
from .parsers import ParsedSection, get_parser, supported_extensions
from .workspace import Workspace


PARSER_VERSION = "1"
CHUNKER_VERSION = "2"  # bumped for contextual retrieval support


# ============================================================================
# Progress reporting
# ============================================================================

@dataclass
class IngestProgress:
    """Snapshot emitted by `ingest()` so UIs can render rich progress."""
    # Current event
    current_path: str = ""
    status: str = ""               # "parsing" | "embedding" | "ok" | "skipped" | "error"
    page: int | None = None
    snippet: str = ""              # 200-char preview of current/last chunk

    # Run-so-far totals (file-level granularity)
    files_done: int = 0
    files_total: int = 0
    bytes_done: int = 0            # bytes of fully-processed files
    bytes_total: int = 0
    chunks_done: int = 0

    # Timing / rate
    elapsed_s: float = 0.0
    eta_s: float | None = None     # remaining seconds, None until rate is known
    rate_bps: float = 0.0          # processed bytes per second (rolling)

    # Index growth
    db_bytes: int = 0              # current size of meta.sqlite on disk

    # Optional payload describing a garbled-page recovery event. When the
    # parser detects garbled extraction and re-runs OCR, it fires one
    # snapshot per recovered page with this populated, so the GUI can
    # render a "before / after" preview of what was fixed in real time.
    # Shape:
    #   {"file": str, "page": int, "image_path": str,
    #    "before": str (≤1500 chars), "after": str (≤1500 chars)}
    recovery: dict | None = None

    @property
    def bytes_pct(self) -> float:
        return (self.bytes_done / self.bytes_total) if self.bytes_total > 0 else 0.0

    @property
    def files_pct(self) -> float:
        return (self.files_done / self.files_total) if self.files_total > 0 else 0.0


# Backward-compat: progress callbacks may take either (path, status)
# or a single IngestProgress.
ProgressCb = Callable[..., None]


@dataclass
class IngestStats:
    files_seen: int = 0
    files_new: int = 0
    files_changed: int = 0
    files_skipped_unchanged: int = 0
    files_unsupported: int = 0
    files_errored: int = 0
    chunks_added: int = 0
    files_removed: int = 0
    seconds: float = 0.0
    errors: list[tuple[str, str]] = None  # type: ignore

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


ProgressCb = Callable[[str, str], None]   # (path, status_text)


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")


def _tokenize(text: str) -> str:
    return " ".join(w.lower() for w in WORD_RE.findall(text))


def _walk_docs(docs_dir: Path) -> list[Path]:
    exts = supported_extensions()
    out: list[Path] = []
    for p in docs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def _emit(progress: ProgressCb | None, snap: IngestProgress) -> None:
    """Send a progress snapshot. Tolerates both signatures —
    `progress(IngestProgress)` and the legacy `progress(path, status)`."""
    if progress is None:
        return
    try:
        progress(snap)
    except TypeError:
        progress(snap.current_path, snap.status)


def _db_size(path: Path) -> int:
    total = 0
    for sib in (path, path.with_suffix(path.suffix + "-wal"),
                path.with_suffix(path.suffix + "-shm")):
        try:
            total += sib.stat().st_size
        except OSError:
            pass
    return total


def ingest(
    ws: Workspace,
    *,
    cfg: Config | None = None,
    force: bool = False,
    progress: ProgressCb | None = None,
) -> IngestStats:
    # ★ FIRST EMIT — happens within microseconds of the call so the UI's
    # "Starting…" placeholder never sits there for more than one frame.
    # We pass None for ws-derived size so it's truly cheap.
    t0 = time.perf_counter()
    if progress is not None:
        try:
            progress(IngestProgress(status="ez-rag ingest waking up…"))
        except TypeError:
            try: progress("", "ez-rag ingest waking up…")
            except Exception: pass

    cfg = cfg or ws.load_config()
    stats = IngestStats()

    bytes_done = 0
    files_done = 0
    chunks_done = 0
    chunks_in_progress = 0
    file_bytes_in_progress = 0

    def _early_snap(status: str, files_total: int = 0, bytes_total: int = 0) -> IngestProgress:
        elapsed = time.perf_counter() - t0
        return IngestProgress(
            status=status,
            files_done=files_done, files_total=files_total,
            bytes_done=bytes_done, bytes_total=bytes_total,
            chunks_done=chunks_done,
            elapsed_s=elapsed,
            db_bytes=_db_size(ws.meta_db_path),
        )

    # ----- "Still working" heartbeat for slow blocking phases ------------
    # Some setup phases (fastembed first-run weight download, big-PDF
    # parse, etc.) block for 10-90 seconds inside a single library call
    # we can't instrument. This heartbeat thread runs in parallel and
    # emits a friendly "still loading X — Ns elapsed" message every 2s
    # until told to stop. Stops cleanly via the stop_evt flag.
    import threading as _t

    def _start_heartbeat(label_fn, stop_evt):
        """label_fn() → current message; called every 2s with elapsed."""
        def _beat():
            n = 0
            while not stop_evt.wait(2.0):
                n += 1
                try:
                    msg = label_fn(n * 2)
                    _emit(progress, _early_snap(msg))
                except Exception:
                    break
        thr = _t.Thread(target=_beat, daemon=True)
        thr.start()
        return thr

    # ----- Pre-ingest narration ------------------------------------------
    _emit(progress, _early_snap(
        "checking which backend you're configured for "
        "(Ollama / llama.cpp / none)…"
    ))
    backend_t0 = time.perf_counter()
    backend = detect_backend(cfg)
    backend_dt = time.perf_counter() - backend_t0
    if backend == "ollama":
        backend_blurb = (
            f"  ↪ Ollama is alive at {cfg.llm_url} "
            f"(probed in {backend_dt*1000:.0f}ms)"
        )
    elif backend == "llama-cpp":
        backend_blurb = "  ↪ llama-cpp-python detected"
    else:
        backend_blurb = (
            "  ↪ no LLM backend reachable — that's fine if Contextual "
            "Retrieval is off (chat will just have no LLM)"
        )
    _emit(progress, _early_snap(backend_blurb))

    will_unload = (
        getattr(cfg, "unload_llm_during_ingest", True)
        and not cfg.enable_contextual
        and backend == "ollama"
    )
    will_contextualize = cfg.enable_contextual and backend != "none"
    SETUP_STEPS = (1 if will_unload else 0) + 4
    step = [0]

    def _step_msg(label: str) -> str:
        step[0] += 1
        return f"[{step[0]}/{SETUP_STEPS}] {label}"

    plan_bits = [
        f"backend={backend}",
        f"embedder={cfg.embedder_model if cfg.embedder_provider == 'fastembed' else cfg.ollama_embed_model}",
        f"contextual={'ON' if cfg.enable_contextual else 'off'}",
        f"chunk_size={cfg.chunk_size}",
        f"top_k={cfg.top_k}",
    ]
    if will_contextualize:
        plan_bits.append("⚠ slow: 1 LLM call per chunk")
    _emit(progress, _early_snap(
        f"plan · {' · '.join(plan_bits)}"
    ))

    if will_unload:
        t = time.perf_counter()
        _emit(progress, _early_snap(_step_msg(
            f"unloading chat LLM '{cfg.llm_model}' from VRAM "
            "(frees ~22 GB for the embedder)…"
        )))
        # Route the unload to the daemon actually hosting the chat model
        # — under multi-GPU, the chat model and the embedder may be on
        # different daemons, so cfg.llm_url isn't always the right URL.
        from .multi_gpu import resolve_url
        unload_ollama_model(
            resolve_url(cfg, cfg.llm_model, role="chat"),
            cfg.llm_model,
        )
        _emit(progress, _early_snap(
            f"  ↪ unloaded in {time.perf_counter() - t:.1f}s"
        ))

    # Embedder load — the most likely place to actually be slow on
    # first run (fastembed downloads ONNX weights). Heartbeat thread
    # emits "still loading…  Ns elapsed" every 2s.
    t = time.perf_counter()
    is_fastembed = (
        cfg.embedder_provider == "fastembed"
        or (cfg.embedder_provider == "auto" and backend != "ollama")
    )
    if is_fastembed:
        emb_label = cfg.embedder_model
        _emit(progress, _early_snap(_step_msg(
            f"loading embedder '{emb_label}' into RAM "
            "(if first run, downloading ~150 MB-1 GB of ONNX weights "
            "from HuggingFace — be patient)…"
        )))
    else:
        emb_label = cfg.ollama_embed_model
        _emit(progress, _early_snap(_step_msg(
            f"loading embedder '{emb_label}' on Ollama — "
            "weights load to GPU VRAM…"
        )))
    stop_evt = _t.Event()
    _start_heartbeat(
        lambda secs: (
            f"  ↪ still loading '{emb_label}'… {secs}s elapsed "
            "(this is normal on first run)"
        ),
        stop_evt,
    )
    try:
        embedder = make_embedder(cfg)
    finally:
        stop_evt.set()
    _emit(progress, _early_snap(
        f"  ↪ embedder ready in {time.perf_counter() - t:.1f}s — "
        f"{embedder.name} produces {embedder.dim}-dimensional vectors"
    ))

    t = time.perf_counter()
    _emit(progress, _early_snap(_step_msg(
        "opening vector index (SQLite + FTS5)…"
    )))
    index = Index(ws.meta_db_path, embed_dim=embedder.dim)
    db_size = _db_size(ws.meta_db_path)
    _emit(progress, _early_snap(
        f"   ↪ done in {time.perf_counter() - t:.1f}s — "
        f"existing index: {db_size/1e6:.1f} MB on disk"
    ))

    t = time.perf_counter()
    _emit(progress, _early_snap(_step_msg(
        f"scanning {ws.docs_dir.name}/ for supported file types…"
    )))
    docs_dir = ws.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    files = _walk_docs(docs_dir)
    stats.files_seen = len(files)
    _emit(progress, _early_snap(
        f"   ↪ done in {time.perf_counter() - t:.1f}s — "
        f"found {len(files)} ingestible file(s)"
    ))

    t = time.perf_counter()
    _emit(progress, _early_snap(_step_msg(
        "measuring file sizes & extension breakdown…"
    )))
    bytes_total = sum(p.stat().st_size for p in files)
    # Quick ext breakdown so user knows what's coming
    ext_counts: dict[str, int] = {}
    for p in files:
        ext_counts[p.suffix.lower()] = ext_counts.get(p.suffix.lower(), 0) + 1
    breakdown = ", ".join(
        f"{n} {e or '(no ext)'}" for e, n in
        sorted(ext_counts.items(), key=lambda kv: -kv[1])
    ) or "no files"
    _emit(progress, _early_snap(
        f"   ↪ {len(files)} files · {bytes_total/1e6:.1f} MB total · "
        f"{breakdown}"
    ))

    # Big-batch warnings — give the user a heads-up if this run will be slow
    if will_contextualize and len(files) > 5:
        _emit(progress, _early_snap(
            f"⚠ Contextual Retrieval is ON: ~1 LLM call per chunk × "
            f"{len(files)} files. Expect this to take a long time. "
            "Cancel and disable Contextual Retrieval if that's not what you want."
        ))
    if backend == "none":
        _emit(progress, _early_snap(
            "ℹ no LLM detected — chat will fall back to retrieval-only "
            "passage display. Ingest doesn't need an LLM unless Contextual "
            "Retrieval is on."
        ))

    _emit(progress, _early_snap(
        "starting per-file processing…",
        files_total=len(files), bytes_total=bytes_total,
    ))

    def snapshot(*, current_path: str = "", status: str = "",
                 page: int | None = None, snippet: str = "") -> IngestProgress:
        elapsed = time.perf_counter() - t0
        # Mid-file in-flight numbers feed the headline bar so it never
        # sits at "0 chunks / 0 B" while a giant file is still grinding.
        live_bytes = bytes_done + file_bytes_in_progress
        live_chunks = chunks_done + chunks_in_progress
        rate = (live_bytes / elapsed) if elapsed > 0 and live_bytes > 0 else 0.0
        eta = ((bytes_total - live_bytes) / rate) if rate > 0 else None
        return IngestProgress(
            current_path=current_path, status=status, page=page,
            snippet=snippet,
            files_done=files_done, files_total=len(files),
            bytes_done=live_bytes, bytes_total=bytes_total,
            chunks_done=live_chunks,
            elapsed_s=elapsed, eta_s=eta, rate_bps=rate,
            db_bytes=_db_size(ws.meta_db_path),
        )

    # Initial emission so UIs can size the bar.
    _emit(progress, snapshot(status="starting"))

    present_rel: set[str] = set()
    for path in files:
        rel = str(path.relative_to(ws.root))
        present_rel.add(rel)
        file_size = path.stat().st_size
        try:
            sha = file_sha256(path)
            existing = index.file_state(rel)
            if (
                not force
                and existing is not None
                and existing.sha256 == sha
            ):
                stats.files_skipped_unchanged += 1
                bytes_done += file_size
                files_done += 1
                _emit(progress, snapshot(current_path=rel, status="unchanged"))
                continue
            _emit(progress, snapshot(current_path=rel, status="parsing"))
            parser = get_parser(path)
            if parser is None:
                stats.files_unsupported += 1
                bytes_done += file_size
                files_done += 1
                continue

            # Throttled per-page callback — keeps the UI alive on big PDFs
            # without spamming page.update() for every page of a 200pp book.
            last_emit_t = [time.perf_counter()]

            def page_cb(page_idx: int, total_pages: int, ocr: bool = False) -> None:
                now = time.perf_counter()
                # Throttle to ~10 fps (was 5) so tickers feel live, but
                # always emit on the last page so the final state lands.
                if (now - last_emit_t[0]) < 0.1 and page_idx != total_pages:
                    return
                last_emit_t[0] = now
                label = "OCR'ing" if ocr else "parsing"
                # Include rate + ETA so the user can see we're not stuck
                # on a hard page (some pages take 5-10× as long as others).
                elapsed = now - parse_started[0]
                rate = page_idx / elapsed if elapsed > 0 else 0.0
                eta_s = ((total_pages - page_idx) / rate) if rate > 0 else 0.0
                eta_str = (f" · ETA {int(eta_s)}s"
                           if 0 < eta_s < 9999 else "")
                _emit(progress, snapshot(
                    current_path=rel,
                    status=(f"{label} page {page_idx}/{total_pages} "
                            f"({rate:.1f} pg/s{eta_str})"),
                    page=page_idx,
                ))

            # Tracked separately from last_emit_t so the rate calc is
            # measured from the start of THIS parse, not the throttle.
            parse_started = [time.perf_counter()]

            # Recovery callback — fires once per page that needed OCR
            # rescue. Builds a snapshot with the recovery payload set so
            # the GUI can show a live before/after card. Only wired up
            # when previews are turned on, otherwise the parser skips
            # the page-render-to-disk step entirely.
            recovery_cb = None
            if getattr(cfg, "preview_garbled_recoveries", False):
                def _on_recovery(payload):
                    snap = snapshot(
                        current_path=rel,
                        status=(f"recovered garbled page {payload.get('page')}"
                                f" via OCR"),
                        page=payload.get("page"),
                    )
                    snap.recovery = {**payload, "kind": "ocr"}
                    _emit(progress, snap)
                recovery_cb = _on_recovery

            parse_t0 = time.perf_counter()
            # When LLM correction is enabled, parse permissively — keep
            # questionable OCR results in the section list so the LLM
            # can take a shot at them instead of having them dropped
            # upstream as TOC fragments / still-garbled.
            permissive = getattr(cfg, "llm_correct_garbled", False)
            parser_kwargs = {"on_progress": page_cb}
            if recovery_cb is not None:
                parser_kwargs["on_recovery"] = recovery_cb
            if permissive:
                parser_kwargs["permissive"] = True
            try:
                sections = parser(path, **parser_kwargs)
            except TypeError:
                # Parser predates one or more kwargs — degrade gracefully.
                for drop in ("permissive", "on_recovery", "on_progress"):
                    parser_kwargs.pop(drop, None)
                    try:
                        sections = parser(path, **parser_kwargs)
                        break
                    except TypeError:
                        continue
                else:
                    sections = parser(path)
            parse_s = time.perf_counter() - parse_t0
            if not sections:
                bytes_done += file_size
                files_done += 1
                _emit(progress, snapshot(
                    current_path=rel,
                    status=f"empty — parser found no extractable text "
                           f"({parse_s:.1f}s)",
                ))
                continue
            # After parse completes, give the user a "what we got" line so
            # they can see the file actually had content before chunking
            # starts. Useful for big PDFs that took minutes to parse.
            sec_summary = f"{len(sections)} section(s)"
            if any(s.page is not None for s in sections):
                pages = sorted({s.page for s in sections if s.page is not None})
                sec_summary += f" across pages {pages[0]}–{pages[-1]}"
            _emit(progress, snapshot(
                current_path=rel,
                status=(f"parsed in {parse_s:.1f}s — {sec_summary} · "
                        f"chunking @ ~{cfg.chunk_size} tokens "
                        f"(overlap {cfg.chunk_overlap})…"),
            ))

            # Optional LLM-assisted quality inspection — second-guesses
            # the parser's heuristic garbled-text detector by asking the
            # LLM to classify each section. Garbled sections are dropped.
            # Slow (1 LLM call per section); off by default.
            if (getattr(cfg, "llm_inspect_pages", False)
                    and detect_backend(cfg) != "none"
                    and len(sections) > 0):
                insp_t0 = time.perf_counter()
                kept: list[ParsedSection] = []
                garbled = 0
                partial = 0
                for si, sec in enumerate(sections, start=1):
                    if not (sec.text or "").strip():
                        kept.append(sec)
                        continue
                    verdict = inspect_text_quality(sec.text, cfg)
                    state = verdict.get("state", "unknown")
                    if state == "garbled":
                        garbled += 1
                        # Drop — don't poison the index. Note in the file's
                        # ingest stream so the user can audit later.
                        continue
                    if state == "partial":
                        partial += 1
                        # Keep partial sections — better partial than nothing.
                        sec.meta = {**(sec.meta or {}), "llm_inspect": "partial"}
                    kept.append(sec)
                    # Emit per-section progress so the user sees motion
                    # during what could be a multi-minute pass on big PDFs.
                    if si == len(sections) or si % 5 == 0:
                        _emit(progress, snapshot(
                            current_path=rel,
                            status=(f"LLM-inspecting section {si}/"
                                    f"{len(sections)} (garbled so far: "
                                    f"{garbled}, partial: {partial})"),
                            page=sec.page,
                        ))
                insp_s = time.perf_counter() - insp_t0
                sections = kept
                _emit(progress, snapshot(
                    current_path=rel,
                    status=(f"LLM inspect done in {insp_s:.1f}s — "
                            f"kept {len(kept)} sections, dropped "
                            f"{garbled} garbled, flagged {partial} partial"),
                ))
                if not sections:
                    bytes_done += file_size
                    files_done += 1
                    _emit(progress, snapshot(
                        current_path=rel,
                        status="all sections rejected by LLM inspect",
                    ))
                    continue

            # Optional LLM-assisted correction of questionable sections.
            # Targets:
            #   - sections that came back from OCR recovery (meta.ocr=True)
            #   - sections the LLM inspect pass flagged as "partial"
            # Each candidate is sent to the LLM with a small surrounding
            # context window for a best-effort cleanup. UNRECOVERABLE
            # responses cause the section to be dropped. Skipped when the
            # LLM backend isn't reachable so this is safe to leave on.
            if (getattr(cfg, "llm_correct_garbled", False)
                    and detect_backend(cfg) != "none"
                    and len(sections) > 0):
                candidates = [
                    si for si, sec in enumerate(sections)
                    if (sec.text or "").strip() and (
                        (sec.meta or {}).get("ocr") is True
                        or (sec.meta or {}).get("llm_inspect") == "partial"
                        or (sec.meta or {}).get("questionable") is True
                    )
                ]
                if candidates:
                    corr_t0 = time.perf_counter()
                    fixed = 0
                    dropped = 0
                    drop_idx: set[int] = set()
                    # When previews are on, also fire a recovery event for
                    # each successful correction so the GUI can show the
                    # LLM-cleaned text as a third panel on the recovery card.
                    preview_on = getattr(cfg, "preview_garbled_recoveries",
                                          False)
                    for n, si in enumerate(candidates, start=1):
                        sec = sections[si]
                        # Build small surrounding context from neighbors
                        # (skipping placeholders / empty meta).
                        before_ctx = sections[si - 1].text if si > 0 else ""
                        after_ctx = (sections[si + 1].text
                                     if si + 1 < len(sections) else "")
                        context = (before_ctx[-400:] + "\n\n"
                                   + after_ctx[:400]).strip()
                        original_text = sec.text
                        cleaned = correct_garbled_text(
                            original_text, cfg, context=context,
                        )
                        if cleaned is None:
                            dropped += 1
                            drop_idx.add(si)
                            if preview_on:
                                snap = snapshot(
                                    current_path=rel,
                                    status=(f"LLM declined to correct page "
                                            f"{sec.page} (kept original)"),
                                    page=sec.page,
                                )
                                snap.recovery = {
                                    "kind": "correction",
                                    "file": rel,
                                    "page": sec.page,
                                    "before": original_text,
                                    "after": "",
                                    "unrecoverable": True,
                                }
                                _emit(progress, snap)
                        else:
                            sec.text = cleaned
                            sec.meta = {**(sec.meta or {}),
                                        "llm_corrected": True}
                            fixed += 1
                            if preview_on:
                                snap = snapshot(
                                    current_path=rel,
                                    status=(f"LLM corrected page {sec.page}"),
                                    page=sec.page,
                                )
                                snap.recovery = {
                                    "kind": "correction",
                                    "file": rel,
                                    "page": sec.page,
                                    "before": original_text,
                                    "after": cleaned,
                                }
                                _emit(progress, snap)
                        if n == len(candidates) or n % 3 == 0:
                            _emit(progress, snapshot(
                                current_path=rel,
                                status=(f"LLM-correcting section {n}/"
                                        f"{len(candidates)} "
                                        f"(fixed {fixed}, dropped {dropped})"),
                                page=sec.page,
                            ))
                    if drop_idx:
                        sections = [s for i, s in enumerate(sections)
                                    if i not in drop_idx]
                    corr_s = time.perf_counter() - corr_t0
                    _emit(progress, snapshot(
                        current_path=rel,
                        status=(f"LLM correction done in {corr_s:.1f}s — "
                                f"fixed {fixed}, dropped {dropped} "
                                f"unrecoverable from {len(candidates)} "
                                f"questionable section(s)"),
                    ))
                    if not sections:
                        bytes_done += file_size
                        files_done += 1
                        _emit(progress, snapshot(
                            current_path=rel,
                            status="all sections rejected by LLM correct",
                        ))
                        continue

            chunk_t0 = time.perf_counter()
            chunks = chunk_sections(
                sections,
                chunk_tokens=cfg.chunk_size,
                overlap_tokens=cfg.chunk_overlap,
            )
            chunk_s = time.perf_counter() - chunk_t0
            if not chunks:
                bytes_done += file_size
                files_done += 1
                _emit(progress, snapshot(current_path=rel, status="no chunks"))
                continue
            _emit(progress, snapshot(
                current_path=rel,
                status=(f"chunked in {chunk_s:.2f}s — {len(chunks)} chunks "
                        f"to embed"),
            ))
            # Pick a representative snippet (early-mid chunk) for live display.
            sample = chunks[min(len(chunks) - 1, len(chunks) // 2)]
            snippet = (sample.text or "").strip().replace("\n", " ")[:240]

            # Optional Anthropic-style Contextual Retrieval.
            # Contextualization is the slowest possible phase (1 LLM call
            # per chunk × hundreds of chunks). Track in-flight chunks +
            # bytes so the headline counters tick smoothly while it runs.
            embed_texts = [c.text for c in chunks]
            if cfg.enable_contextual and detect_backend(cfg) != "none":
                full_text = "\n\n".join(s.text for s in sections)
                ctx_t0 = time.perf_counter()
                ctx_texts: list[str] = []
                for ci, c in enumerate(chunks, start=1):
                    ctx_texts.append(
                        contextualize_chunk(c.text, full_text, cfg)
                    )
                    # Tick the in-flight counters every chunk so headline
                    # numbers move even though the file isn't committed.
                    chunks_in_progress = ci
                    file_bytes_in_progress = int(
                        file_size * (ci / len(chunks)) * 0.5  # ctx is half the work; embed is the other half
                    )
                    elapsed = time.perf_counter() - ctx_t0
                    rate = ci / elapsed if elapsed > 0 else 0.0
                    eta = ((len(chunks) - ci) / rate) if rate > 0 else 0.0
                    if eta >= 60:
                        eta_str = f"{int(eta // 60)}m{int(eta % 60):02d}s"
                    else:
                        eta_str = f"{int(eta)}s"
                    # Emit every chunk — at ~3s per chunk for a reasoning
                    # model the UI desperately needs the tick. The watchdog
                    # also fires at 0.5s so the elapsed clock keeps moving
                    # between chunks.
                    # Refresh the snippet to whatever's currently being
                    # contextualized so the snippet card doesn't sit on
                    # a stale page-113 sample for an entire 30-min run.
                    cur_snippet = (c.text or "").strip().replace(
                        "\n", " ")[:240]
                    _emit(progress, snapshot(
                        current_path=rel,
                        status=(f"contextualizing chunk {ci}/"
                                f"{len(chunks)} (ETA {eta_str})"),
                        page=c.page, snippet=cur_snippet,
                    ))
                embed_texts = ctx_texts
            _emit(progress, snapshot(
                current_path=rel,
                status=f"embedding {len(chunks)} chunks",
                page=sample.page, snippet=snippet,
            ))
            # Embed in mini-batches so the UI updates during long embedding
            # passes (big PDFs with hundreds of chunks). For tiny files this
            # is still effectively a single call.
            BATCH = max(1, getattr(cfg, "embed_batch_size", 16))
            import numpy as np
            vec_chunks = []
            for i in range(0, len(embed_texts), BATCH):
                batch = embed_texts[i : i + BATCH]
                vec_chunks.append(np.asarray(embedder.embed(batch),
                                             dtype=np.float32))
                # Embedding is the second half of "in-progress" — count
                # from 0.5 to 1.0 of file_size so the bar smoothly fills
                # whether or not contextualization ran.
                processed = min(i + BATCH, len(embed_texts))
                base_pct = 0.5 if cfg.enable_contextual else 0.0
                file_bytes_in_progress = int(file_size * (
                    base_pct + (1 - base_pct) * (processed / len(embed_texts))
                ))
                chunks_in_progress = processed
                # Always emit per batch (was: only when len > BATCH) — even
                # tiny files benefit from a "embedding 4/4 chunks" tick.
                _emit(progress, snapshot(
                    current_path=rel,
                    status=f"embedding {processed}/{len(embed_texts)} chunks",
                    page=sample.page, snippet=snippet,
                ))
            vecs = np.concatenate(vec_chunks, axis=0) if vec_chunks else \
                   np.zeros((0, embedder.dim), dtype=np.float32)

            # Per-file metadata FTS5 boost: if a sidecar exists for
            # this file, prepend matching entity terms to the chunk's
            # `tokens` column so BM25 lights up on entity names even
            # when the user's query phrasing differs. This is a free
            # quality lift — costs nothing at retrieval time, costs
            # one TOML read here at ingest. Failures are silent.
            entities_for_file: list[str] = []
            priority_for_file: list[str] = []
            try:
                if getattr(cfg, "use_file_metadata", True):
                    from .ingest_meta import load as _load_meta
                    md = _load_meta(path, workspace_root=ws.root)
                    if md is not None:
                        entities_for_file = md.entities.all()
                        priority_for_file = list(md.priority_terms)
            except Exception:
                # Never let a sidecar parse error block ingest.
                pass
            entity_set = entities_for_file + priority_for_file

            rows = []
            for c, v in zip(chunks, vecs):
                tokens = _tokenize(c.text)
                # Inject matching entities into the FTS tokens. Cheap
                # case-insensitive substring scan against the chunk's
                # text. We only inject entities ALREADY PRESENT in the
                # chunk (no false positives), so BM25 still respects
                # locality.
                if entity_set:
                    text_lower = (c.text or "").lower()
                    matched: list[str] = []
                    for term in entity_set:
                        if term and term.lower() in text_lower:
                            matched.append(term)
                    if matched:
                        # Lowercase + dedupe for the FTS5 column.
                        seen: set[str] = set()
                        boost: list[str] = []
                        for t in matched:
                            tl = t.lower()
                            if tl in seen:
                                continue
                            seen.add(tl)
                            boost.append(tl)
                        tokens = tokens + " " + " ".join(boost)
                rows.append((c.ord, c.page, c.section, c.text, tokens, v))

            # Chapter boundaries (PDF outline / heading sections / fallback
            # 'Document'). Cheap — pypdf already opened the file in the
            # parser cache. Failures here must not block ingest.
            try:
                chapters = detect_chapters(path, chunks)
            except Exception:
                chapters = []

            index.replace_file(
                path=rel,
                sha256=sha,
                bytes_=file_size,
                mtime=path.stat().st_mtime,
                parser_version=PARSER_VERSION,
                chunker_version=CHUNKER_VERSION,
                embedder=embedder.name,
                chunks=rows,
                chapters=chapters,
            )
            stats.chunks_added += len(rows)
            chunks_done += len(rows)
            if existing is None:
                stats.files_new += 1
            else:
                stats.files_changed += 1
            bytes_done += file_size
            files_done += 1
            # File committed — clear the in-flight counters so the next
            # file's progress is measured from zero again (and we don't
            # double-count what we just committed into chunks_done).
            chunks_in_progress = 0
            file_bytes_in_progress = 0
            _emit(progress, snapshot(
                current_path=rel,
                status=f"ok ({len(rows)} chunks)",
                page=sample.page, snippet=snippet,
            ))
        except Exception as e:
            stats.files_errored += 1
            stats.errors.append((str(path), repr(e)))
            bytes_done += file_size
            files_done += 1
            chunks_in_progress = 0
            file_bytes_in_progress = 0
            _emit(progress, snapshot(current_path=rel,
                                     status=f"ERROR: {e}"))

    stats.files_removed = index.delete_missing(present_rel)
    stats.seconds = time.perf_counter() - t0
    _emit(progress, snapshot(status="done"))
    return stats

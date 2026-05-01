"""Download free / public-domain corpora for ez-rag testing.

Each corpus pulls a curated set of Wikipedia (CC BY-SA 4.0) articles as
HTML files into the chosen folder. HTML is preferred over plain text so
ez-rag's chapter-aware retrieval (built on heading detection) has
something to work with.

Usage:
    python bench/fetch_test_corpus.py dinosaurs   ./test-rags/dinosaurs
    python bench/fetch_test_corpus.py mythology   ./test-rags/mythology
    python bench/fetch_test_corpus.py presidents  ./test-rags/presidents
    python bench/fetch_test_corpus.py space       ./test-rags/space

Wikipedia content is licensed CC BY-SA 4.0 — fine for personal testing
and for redistributing if you preserve attribution. We embed the source
URL at the top of each saved HTML file as a comment so the licence trail
is intact.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.parse import quote

import httpx


WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/html/{title}"
USER_AGENT = (
    "ez-rag-test-corpus/0.1 (https://github.com/jrounds99/ez-rag; "
    "personal testing only)"
)


# ---------------------------------------------------------------------------
# Curated topic lists. Each is "interesting enough that retrieval is
# non-trivial" + "has clear factual content that's easy to ask questions
# about". Sized for quick ingest (~20–30 articles per topic).
# ---------------------------------------------------------------------------

CORPORA = {
    # Classic test corpus — recognizable names, factual content (size,
    # diet, period, fossil locations) that's easy to ask retrieval
    # questions about. Mostly stub-to-medium articles.
    "dinosaurs": [
        "Tyrannosaurus", "Triceratops", "Apatosaurus", "Allosaurus",
        "Stegosaurus", "Brachiosaurus", "Diplodocus", "Iguanodon",
        "Velociraptor", "Deinonychus", "Ceratosaurus", "Ankylosaurus",
        "Parasaurolophus", "Spinosaurus", "Carnotaurus", "Giganotosaurus",
        "Archaeopteryx", "Compsognathus", "Dilophosaurus", "Corythosaurus",
        "Edmontosaurus", "Oviraptor", "Troodon", "Eoraptor",
        "Pachycephalosaurus", "Pteranodon", "Mosasaurus",
        "Cretaceous–Paleogene_extinction_event",
        "Mesozoic", "Dinosaur",
    ],
    # Greek mythology — names with overlapping relationships, tests how
    # well retrieval handles entity disambiguation.
    "mythology": [
        "Zeus", "Hera", "Poseidon", "Hades", "Athena", "Apollo",
        "Artemis", "Ares", "Aphrodite", "Hermes", "Hephaestus",
        "Demeter", "Dionysus", "Heracles", "Perseus_(Greek_mythology)",
        "Theseus", "Achilles", "Odysseus", "Jason", "Orpheus",
        "Medusa", "Minotaur", "Pegasus", "Cyclopes", "Titans",
        "Iliad", "Odyssey", "Trojan_War",
    ],
    # US Presidents — every article has the same structure (early life /
    # presidency / legacy), good for testing chapter-aware retrieval.
    "presidents": [
        "George_Washington", "John_Adams", "Thomas_Jefferson",
        "James_Madison", "James_Monroe", "Andrew_Jackson",
        "Abraham_Lincoln", "Ulysses_S._Grant", "Theodore_Roosevelt",
        "Woodrow_Wilson", "Franklin_D._Roosevelt", "Harry_S._Truman",
        "Dwight_D._Eisenhower", "John_F._Kennedy", "Lyndon_B._Johnson",
        "Richard_Nixon", "Jimmy_Carter", "Ronald_Reagan",
        "George_H._W._Bush", "Bill_Clinton", "George_W._Bush",
        "Barack_Obama",
    ],
    # Space exploration — long technical articles with lots of dates,
    # numbers, missions. Stresses retrieval precision.
    "space": [
        "Apollo_program", "Apollo_11", "Apollo_13", "Saturn_V",
        "International_Space_Station", "Hubble_Space_Telescope",
        "James_Webb_Space_Telescope", "Mars_rover", "Curiosity_(rover)",
        "Perseverance_(rover)", "Voyager_program", "Voyager_1",
        "Voyager_2", "Cassini–Huygens", "New_Horizons",
        "Space_Shuttle", "SpaceX_Dragon", "Falcon_9", "Starship",
        "Artemis_program", "Mariner_program",
    ],
}


def fetch(title: str, *, client: httpx.Client) -> str:
    """Pull one Wikipedia article as HTML. Raises on HTTP error."""
    url = WIKI_API.format(title=quote(title, safe=""))
    r = client.get(url, follow_redirects=True, timeout=30.0)
    r.raise_for_status()
    src = f"https://en.wikipedia.org/wiki/{title}"
    # Prepend a comment so the CC BY-SA attribution travels with the file.
    header = (
        f"<!-- Source: {src}\n"
        f"     Licence: CC BY-SA 4.0\n"
        f"     Fetched: {time.strftime('%Y-%m-%d %H:%M:%S')} -->\n"
    )
    return header + r.text


def main():
    if len(sys.argv) < 2:
        print("Usage: fetch_test_corpus.py <topic> [<dest>]")
        print(f"Topics: {', '.join(CORPORA)}")
        return 1
    topic = sys.argv[1]
    if topic not in CORPORA:
        print(f"Unknown topic '{topic}'. Available: {', '.join(CORPORA)}")
        return 1
    dest = Path(sys.argv[2] if len(sys.argv) > 2 else f"./test-rags/{topic}")
    dest.mkdir(parents=True, exist_ok=True)

    titles = CORPORA[topic]
    print(f"Fetching {len(titles)} '{topic}' articles -> {dest.resolve()}")
    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        ok, fail = 0, 0
        for i, title in enumerate(titles, 1):
            out = dest / f"{title}.html"
            if out.exists() and out.stat().st_size > 1000:
                print(f"  [{i:>2}/{len(titles)}] skip (cached): {title}")
                ok += 1
                continue
            try:
                html = fetch(title, client=client)
                out.write_text(html, encoding="utf-8")
                kb = out.stat().st_size / 1024
                print(f"  [{i:>2}/{len(titles)}] {title:<45} {kb:>6.1f} KB")
                ok += 1
                # Be polite — Wikipedia REST allows ~200 req/s but we
                # don't need to hammer it.
                time.sleep(0.15)
            except Exception as e:
                print(f"  [{i:>2}/{len(titles)}] FAIL  {title}: {e}")
                fail += 1
        total_size = sum(p.stat().st_size for p in dest.glob("*.html"))
        print(f"\nDone: {ok} ok, {fail} fail, {total_size/1e6:.1f} MB total")
        print(f"\nNext: open ez-rag and create a new RAG pointing at  {dest}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

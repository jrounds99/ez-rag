# Bench test corpus

Tiny public-domain corpus that ships with the bench so a fresh `git
clone` can run `bench/run.sh` and produce a complete result bundle
without the user supplying any documents.

## Files

| File | Source | Words |
|---|---|---|
| `programming_pearls.md` | Custom synthetic article (CC0) — describes algorithmic problem-solving | ~600 |
| `linnaean_taxonomy.md` | Custom synthetic article (CC0) — biology classification system | ~500 |
| `the_yellow_wallpaper_excerpt.md` | Charlotte Perkins Gilman, 1892 (public domain) — short excerpt | ~400 |
| `held_out_questions.jsonl` | Hand-curated retrieval-recall questions, one per topic | — |

The corpus is deliberately **diverse** (technical / scientific /
literary) so the recall benchmarks exercise different vocabulary
distributions. It's also deliberately **small** (~5 KB total) so
ingest takes seconds, not minutes — the bench is verifying behavior,
not stress-testing throughput.

## Replacing it

Pass `--corpus PATH/TO/YOUR/CORPUS` to any `ez-rag-bench` subcommand
to use a different folder. If your corpus has its own
`held_out_questions.jsonl`, the recall@k metrics will use it.
Otherwise the recall metrics are skipped and the report shows "—".

# CLI reference

```
ez-rag init [PATH]                       Create a workspace.
ez-rag ingest [--force] [--watch]        Parse, chunk, embed everything in docs/.
ez-rag reindex                           Drop and rebuild the index.
ez-rag ask "question" [--top-k N]
       [--json] [--no-citations] [--no-hybrid]
                                         One-shot Q&A.
ez-rag chat [--top-k N]                  Interactive REPL.
ez-rag status                            Workspace stats.
ez-rag models                            Show LLM + embedder configuration.
ez-rag serve [--host H] [--port P]       OpenAI-compatible /v1/chat/completions.
ez-rag doctor                            Environment diagnostics.
ez-rag help <topic>                      This manual.
```

Topics: `getting-started`, `ingestion`, `models`, `chat`, `gui`, `ocr`, `cli`.

## --json output for `ask`

```json
{
  "question": "…",
  "answer": "…",
  "backend": "ollama",
  "retrieved": [
    {"doc_id":"docs/foo.pdf","source":"docs/foo.pdf","page":3,
     "section":"Introduction","score":0.81,"text":"…"}
  ]
}
```

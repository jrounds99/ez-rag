# Chat

ez-rag's chat is a conversation, not single-turn Q&A.

## Behavior

- Each new message is sent to the LLM **with the full prior conversation** plus retrieved passages for the latest question.
- The LLM is told to answer from the retrieved context when possible (with `[1]`, `[2]`, … citations) and to chat normally otherwise.
- "What about the second one?" / "explain that more simply" / "can you re-summarize" work because the model has the prior turns in context.

## In the GUI

```
ez-rag-gui
```

- **Enter** sends, **Shift+Enter** inserts a newline.
- The send arrow turns into a red ⊘ Stop button while the model is generating.
- Citation chips below each grounded answer expand when clicked, showing the exact retrieved passage.
- **Clear** in the chat header starts a fresh conversation.

## In the CLI

```bash
ez-rag chat                     # interactive REPL
ez-rag ask "your question"      # one-shot, prints answer + citations
```

`chat` keeps a session in memory; quit with `/exit`. `ask` is single-turn.

For scripting, `--json` returns structured retrieval + answer:

```bash
ez-rag ask "What does the corpus say?" --json | jq .
```

## OpenAI-compatible HTTP

```bash
ez-rag serve --port 11533
```

Then point any OpenAI SDK at `http://127.0.0.1:11533/v1`. The response includes
an extra `ez_rag_citations` array with the retrieved sources.

## What controls quality

- **Retrieval** — `top_k` and `hybrid` in Settings. Higher top-K = more context but slower and noisier; hybrid (BM25 + dense) usually beats either alone.
- **Model size** — bigger ≈ better, until VRAM runs out. See [models](models) for picks per VRAM tier.
- **Chunk size** — too small loses context, too large dilutes match scores. 512 tokens with 64-token overlap is a solid default.
- **Contextual Retrieval** — Anthropic-style chunk-context summaries before embedding. Off by default; toggle in Settings. Slower to ingest, materially better recall on technical/structured docs.

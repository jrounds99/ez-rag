# Examples

End-to-end recipes. Each one is self-contained.

## 1. PDFs in a folder

```bash
mkdir research && cd research
ez-rag init .
cp ~/papers/*.pdf docs/
ez-rag ingest

ez-rag ask "Which papers discuss attention mechanisms?"
ez-rag ask "Summarize the main contribution of each paper" --top-k 20
```

## 2. Screenshots / image dumps

```bash
mkdir notes && cd notes
ez-rag init .
cp ~/Pictures/screenshots/*.png docs/
# install OCR support if you don't have it:
pipx inject ez-rag rapidocr-onnxruntime Pillow
ez-rag ingest
ez-rag ask "What was the error message in any of the screenshots?"
```

## 3. Mixed corpus (PDFs + DOCX + spreadsheets + HTML)

```bash
mkdir handbook && cd handbook
ez-rag init .
cp ~/handbook/*.pdf docs/
cp ~/handbook/*.docx docs/
cp ~/handbook/*.xlsx docs/
cp ~/handbook/*.html docs/
ez-rag ingest
ez-rag chat
```

`ez-rag` ranks by hybrid BM25 + dense similarity, so a question about something
that lives only in a spreadsheet finds it even if the wording differs.

## 4. Live ingest while you drop files in

In one terminal:

```bash
ez-rag ingest --watch
```

In another, drop files into `docs/`. They are picked up within ~2 seconds.

## 5. OpenAI-compatible HTTP endpoint

```bash
ez-rag serve --host 0.0.0.0 --port 11533
# in another shell:
curl -s http://127.0.0.1:11533/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:7b-instruct","messages":[{"role":"user","content":"What does the corpus say about X?"}]}' | jq .
```

Use it from any OpenAI SDK by pointing the base URL at
`http://127.0.0.1:11533/v1`.

## 6. Air-gapped laptop

1. On a connected machine: `ollama pull qwen2.5:7b-instruct`, `ollama pull nomic-embed-text`. Copy `~/.ollama/models` to the laptop.
2. `pipx install ez-rag` and copy any wheels onto the laptop, install offline.
3. On the laptop: `ez-rag init && ez-rag ingest && ez-rag chat`.

ez-rag never phones home: no telemetry, no remote API. Once the wheels and
Ollama models are in place, everything runs offline.

## 7. Use a local GGUF directly (no Ollama)

```bash
pipx inject ez-rag llama-cpp-python
# get a GGUF, e.g. Qwen2.5-7B-Instruct-Q4_K_M.gguf
# then edit .ezrag/config.toml:
```

```toml
llm_provider = "llama-cpp"
llm_model = "/abs/path/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
```

```bash
ez-rag chat
```

## 8. GUI

```bash
pipx inject ez-rag flet      # if not installed
ez-rag-gui
```

Click "Open workspace…", pick or create a folder, drop files in, hit
"Ingest", and start chatting in the Chat tab.

# Third-Party Notices

ez-rag is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).
It depends on third-party open-source software. This file documents the
licenses and attributions for those dependencies, with particular care
for the **optional** components that ship behind extras.

ez-rag does **not** vendor or redistribute the source of these packages;
they are installed from PyPI as declared dependencies. This file is
provided for transparency and to honor the upstream attribution
requests, including those carried in each project's own `NOTICE` file.

---

## Optional: context compression — `ez-rag[compress]`

The optional context-compression feature
(see [docs/COMPRESSION.md](docs/COMPRESSION.md)) is **off by default**
and only active when the user installs `ez-rag[compress]` and sets
`compress_context = true`. It is powered by:

### headroom-ai

- **Project:** Headroom — https://github.com/chopratejas/headroom
- **License:** Apache License 2.0
- **Copyright:** Copyright 2025 Headroom Contributors

Headroom's own `NOTICE` file attributes the following libraries, which
we reproduce here per Apache-2.0 §4(d):

> Headroom
> Copyright 2025 Headroom Contributors
>
> This product includes software developed by third parties:
> - **tiktoken** — Copyright OpenAI and Shantanu Jain — MIT License
>   (https://github.com/openai/tiktoken)
> - **Pydantic** — Copyright Pydantic Services Inc. and contributors —
>   MIT License (https://github.com/pydantic/pydantic)
> - **sentence-transformers** — Copyright Nils Reimers — Apache License
>   2.0, with additional model-specific restrictions
>   (https://github.com/UKPLab/sentence-transformers)
> - **FastAPI** — Copyright Sebastián Ramírez — MIT License
>   (https://github.com/fastapi/fastapi)
> - **NumPy** — Copyright NumPy Developers — BSD 3-Clause License
>   (https://github.com/numpy/numpy)

### Runtime dependencies pulled by `ez-rag[compress]`

The compression relevance model runs on these (installed transitively
by headroom-ai's `[ml]` path):

| Package | License | Project |
|---|---|---|
| `torch` | BSD-3-Clause | https://github.com/pytorch/pytorch |
| `transformers` | Apache-2.0 | https://github.com/huggingface/transformers |
| `tiktoken` | MIT | https://github.com/openai/tiktoken |
| `answerdotai/ModernBERT-base` (model weights) | Apache-2.0 | https://huggingface.co/answerdotai/ModernBERT-base |

### Network / privacy note

The compression path runs locally. The only outbound network request it
can make is a **one-time download of the tiktoken tokenizer vocabulary**
(`o200k_base`, ~2.5 MB, from `openaipublic.blob.core.windows.net`) used
for token counting, which tiktoken then caches permanently. **No prompt
text, document content, or user data is transmitted.** To run fully
offline, pre-seed the tiktoken cache and set `TIKTOKEN_CACHE_DIR`, or
keep the cache from a prior online run. See
[docs/COMPRESSION.md](docs/COMPRESSION.md#privacy--network-behavior).

---

## Core dependencies

ez-rag's core (always installed) builds on:

| Package | License |
|---|---|
| `typer`, `click` | MIT / BSD-3-Clause |
| `rich` | MIT |
| `pydantic`, `pydantic-settings` | MIT |
| `platformdirs` | MIT |
| `httpx` | BSD-3-Clause |
| `numpy` | BSD-3-Clause |
| `pypdf` | BSD-3-Clause |
| `python-docx` | MIT |
| `openpyxl` | MIT |
| `beautifulsoup4` | MIT |
| `lxml` | BSD-3-Clause |
| `fastembed` | Apache-2.0 |
| `tqdm` | MPL-2.0 / MIT |
| `psutil` | BSD-3-Clause |

## Other optional extras

| Extra | Key packages | Licenses |
|---|---|---|
| `ocr` | `rapidocr-onnxruntime`, `Pillow`, `pytesseract`, `pypdfium2` | Apache-2.0 / HPND / Apache-2.0 / BSD-3/Apache |
| `gui` | `flet` | Apache-2.0 |
| `llm` | `llama-cpp-python` | MIT |
| `compress` | `headroom-ai` (+ `torch`, `transformers`) | Apache-2.0 (+ BSD / Apache-2.0) |
| `dev` | `pytest`, `pytest-timeout` | MIT |

---

*This notice is maintained on a best-effort basis. License identifiers
follow each project's declared license at the time of writing; consult
each upstream project for the authoritative and current terms.*

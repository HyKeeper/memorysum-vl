# memorysum-vl

**A self-hosted multimodal memory service for agents.** `Add` conversations
(text + screenshots). `Search` them with questions (text, images, or both).
One Python process, one SQLite file, three small models served locally by
[llama.cpp](https://github.com/ggml-org/llama.cpp) — no vector database, no
external API, no data leaving the machine.

[中文说明](README.zh-CN.md)

![python](https://img.shields.io/badge/python-3.12%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-22%20passing-brightgreen)
![models](https://img.shields.io/badge/models-local%20llama.cpp-black)

```
POST /v1/add       messages + images ─┐
                                      ├─► SQLite (text · images · FTS5 · sqlite-vec)
POST /v1/search    question + images ─┘        ▲            ▲             ▲
                                               │            │             │
                                        Rax-4.5      Qwen3-VL-Embedding-2B
                                     (image → text)     (text + image vectors)
                                                            │
                                                    bge-reranker-v2-m3
                                                     (cross-encoder rerank)
```

## Why it exists

Agent memory benchmarks (and real assistants) fail on the same thing: the answer
is often a *screenshot*, and it is buried in a conversation from three weeks ago.
Most memory implementations pick one side — keyword search over text, or a vector
DB over chunks. This project keeps both and lets them argue:

* **Images become text, but pixels are never thrown away.** Every image is
  captioned once (cached by `sha256`) and the original bytes stay in the DB, so a
  result can hand the upstream answer model the actual picture.
* **Seven-ish retrieval routes, fused by rank.** Three lexical routes (words,
  character trigrams, Chinese segmentation), a dense-text and a dense-image route
  per query vector, plus a per-option route for multiple-choice questions.
  Reciprocal Rank Fusion makes their incompatible score scales irrelevant.
* **A cross-encoder decides the head.** `bge-reranker-v2-m3` re-scores the fused
  head; its rank is then blended back with the RRF rank, so multi-route consensus
  still counts.
* **Calibrated scores.** Results carry `sigmoid(reranker_logit)`, so `0.02` means
  "nothing matched" — which is what makes abstention (`data: []`) possible.
* **Explicit degradation.** Kill any model server and `Add`/`Search` still return
  `200`; the affected units are marked `caption_state` / `embedding_state`
  `missing` instead of silently pretending everything is fine.

## Models

All three run locally through llama.cpp on a single 16 GB GPU (~12.4 GB resident).

| Role | Model | Weights | GGUF used here |
|---|---|---|---|
| Image → text | **Rax-4.5** (1.9B VLM, `qwen35` arch) | [raxcore-dev/Rax-4.5](https://huggingface.co/raxcore-dev/Rax-4.5) | [mradermacher/Rax-4.5-GGUF](https://huggingface.co/mradermacher/Rax-4.5-GGUF) · `Rax-4.5.Q8_0.gguf` + `Rax-4.5.mmproj-f16.gguf` |
| Embeddings | **Qwen3-VL-Embedding-2B** (2048-d, shared text+image space) | [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | [DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF](https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF) · `Qwen.Qwen3-VL-Embedding-2B.f16.gguf` + `mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf` |
| Reranking | **bge-reranker-v2-m3** (cross-encoder) | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | [gpustack/bge-reranker-v2-m3-GGUF](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF) (Q8_0; f16 used here) |

Runtime: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (CUDA build).

File sizes, exact launch flags, one-line download commands, and the five real
failure modes (missing `mmproj`, randomised media marker, image-token overflow,
reasoning-model empty `content`, HF mirror): **[docs/models.md](docs/models.md)**.

## Quickstart

Prerequisites: Linux, Python 3.12+, a 16 GB GPU, and a CUDA build of `llama.cpp`.

```bash
# 1. code + python env
git clone https://github.com/HyKeeper/memorysum-vl.git
cd memorysum-vl
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# 2. weights (≈ 6.6 GB, skipped if present)
export MODEL_DIR=~/models
./deploy/llama/download-models.sh

# 3. three model servers
./deploy/llama/start-reranker.sh  &
./deploy/llama/start-captioner.sh &
./deploy/llama/start-embedder.sh   &

# 4. config
cp .env.example .env
mkdir -p .secrets
python -c "import secrets; print(secrets.token_urlsafe(24))" > .secrets/api_token
chmod 600 .secrets/api_token

# 5. run
./run.sh start && ./run.sh status
```

```bash
TOKEN=$(cat .secrets/api_token)

curl -s -X POST localhost:8003/v1/add \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"request_id":"r1","user_id":"u1","session_id":"s1",
       "messages":[{"role":"user","content":"the spare key is in the blue drawer"}]}'

curl -s -X POST localhost:8003/v1/search \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","query":"where are the keys?","top_k":5}'
```

Docker, systemd, tuning knobs, backup and troubleshooting:
**[docs/deployment.md](docs/deployment.md)**.

## API at a glance

| Method | Path | Auth | |
|---|---|---|---|
| `POST` | `/v1/add` | Bearer | `{request_id, user_id, session_id, messages[]}` → `{success: true, ...}` |
| `POST` | `/v1/search` | Bearer | `{user_id, query, top_k, options?}` → `{data: [{id, content, score, created_at}]}` |
| `GET` | `/healthz` | — | `{status: "ok", track: "multimodal"}` |

Full contract, field constraints and error codes: **[docs/api.md](docs/api.md)**.
Run `examples/quickstart.sh` for a copy-paste session, or `examples/client.py`
for a stdlib-only Python client (including a base64 image query).

## How a search actually runs

```
query text + images
  → query images captioned (cached)  →  abstain check
  → lexical: fts_word · fts_trigram · fts_zh · options
  → dense  : dense_text / dense_image for query text, each image, text+image
  → RRF (adaptive k, per-route weights)  → top `max(rerank_top_n, top_k)`
  → bge-reranker-v2-m3 over the fused head
  → rerank rank blended back with RRF rank → sigmoid score
  → weak-evidence clamp · neighbour messages · session labels · images
  → top_k
```

Details, storage schema and the reasoning behind each step:
**[docs/architecture.md](docs/architecture.md)**.

## Project layout

```
src/memorysum_vl/
├── api/        FastAPI routes, bearer auth, ASGI request logging, mappers
├── services/   AddService / SearchService — orchestration only
├── domain/     memory models + ports (MemoryStore, Embedder, Captioner, Reranker)
├── adapters/   sqlite_store, repository (FTS5 + sqlite-vec), three llama.cpp clients
└── core/       settings, DI container, errors, logging, RRF, CJK text, imaging
deploy/         llama.cpp launchers, Dockerfile, compose and systemd examples
docs/           architecture · models · api · deployment
examples/       curl and Python clients
scripts/        smoke_api.py (end-to-end probe), backfill_transcripts.py (repair)
tests/          end-to-end tests with fake model servers (no GPU needed)
```

Hexagonal on purpose: `domain/ports.py` is the only seam the services know about,
so swapping SQLite for Postgres, or any model for another, does not touch the API.

## Development

```bash
make install     # editable install with dev extras
make test        # pytest — 22 end-to-end tests, fake model servers, no GPU
make lint        # ruff check
make fmt         # ruff format
make run         # foreground uvicorn on :8003
```

Tests never call a model: `tests/conftest.py` provides an in-process fake
embedder/captioner/reranker, so the whole Add→Search path (including the
degradation paths) runs in under a second.

## Measured numbers

RTX 5060 Ti 16 GB, one process, three model servers (numbers from the development
machine; treat them as order-of-magnitude, not a benchmark):

| Scenario | Result |
|---|---|
| `Search`, `top_k=20`, warm | **1.5–2.0 s** per request |
| `Search`, `top_k=100`, 16 concurrent clients | p50 ≈ 37 s (queueing on the reranker, GPU-bound) |
| `Add`, 1 image, 3 caption slots | ≈ 7 s |
| Corpus used during evaluation | 499k memory units, 56k images, 30 GB SQLite |
| Retrieval quality on a 164-question internal set | evidence recall 33/36 multi-hop, 47/49 single-hop |

## Limitations

* **Single-writer storage.** SQLite + WAL is wonderful up to a few hundred
  million tokens and then it is not. The store is behind a port, so a Postgres /
  `pgvector` adapter is the natural next step.
* **One track at a time.** `MEMORYSUM_TRACK` selects the contract
  (`multimodal` / `text` / `code`); a single deployment serves one of them.
* **Caption quality is the recall ceiling.** A 1.9B VLM cannot transcribe a dense
  document perfectly; `scripts/backfill_transcripts.py` exists precisely because captions
  are lossy and worth re-doing.
* **No auth model beyond a shared bearer token** (see
  [deployment §9](docs/deployment.md#9-security-notes)).

## License

MIT — see [LICENSE](LICENSE). © 2026 HyKeeper.

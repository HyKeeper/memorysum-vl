# Deployment

## 1. Requirements

| Item | Minimum | Notes |
|---|---|---|
| OS | Linux (x86-64) | developed on Ubuntu; macOS works on CPU only |
| GPU | **16 GB VRAM** | the three models sit at ~12.4 GB resident, leaving headroom |
| Python | **3.12+** | `pyproject.toml` `requires-python` |
| Disk | ~8 GB models + room for the DB | the DB grows with your corpus (see §7) |
| Runtime | [llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA | needs a recent build (Rax-4.5 uses the `qwen35` architecture) |

No GPU? Everything still works on CPU with `-ngl 0`, but captioning and
embedding become seconds-per-image and search latency follows.

## 2. Install

```bash
git clone https://github.com/HyKeeper/memorysum-vl.git
cd memorysum-vl

python3.12 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e '.[dev]'      # omit [dev] for a runtime-only install

cp .env.example .env
# generate a token and point .env at it (see §4)
python -c "import secrets; print(secrets.token_urlsafe(24))" > .secrets/api_token
chmod 600 .secrets/api_token
```

> `Settings` reads `.env` **from the current working directory**, so always start
> the service from the repository root (or pass absolute paths).

## 3. Start the model servers

```bash
export MODEL_DIR=~/models                      # where the GGUFs live
./deploy/llama/download-models.sh              # skips files that already exist
./deploy/llama/start-reranker.sh   &           # :7997  bge-reranker-v2-m3
./deploy/llama/start-captioner.sh  &           # :7996  Rax-4.5
./deploy/llama/start-embedder.sh   &           # :7998  Qwen3-VL-Embedding-2B

curl -s localhost:7996/health; curl -s localhost:7997/health; curl -s localhost:7998/health
```

Model files, sizes, links and the flags that matter: **[models.md](models.md)**.
The scripts pin `LLAMA_MEDIA_MARKER='<__media__>'`, which the client requires —
llama.cpp otherwise randomises the multimodal placeholder on every start.

## 4. Configure

Everything is `MEMORYSUM_*` environment variables (or `.env`). The full list lives
in `src/memorysum_vl/core/config.py`; `.env.example` documents the ones you are
likely to touch. The essentials:

| Variable | Default | Meaning |
|---|---|---|
| `MEMORYSUM_AUTH_TOKEN_FILE` | — | path to a file holding the bearer token (recommended) |
| `MEMORYSUM_AUTH_TOKEN` | — | inline token; use one or the other |
| `MEMORYSUM_DB_PATH` | `data/memorysum.db` | SQLite file; created on first start |
| `MEMORYSUM_TRACK` | `multimodal` | `multimodal` \| `text` \| `code` |
| `MEMORYSUM_API_PREFIX` | `/v1` | route prefix |
| `MEMORYSUM_EMBEDDING_BASE_URL` | `http://127.0.0.1:7998` | embedder |
| `MEMORYSUM_CAPTION_BASE_URL` | `http://127.0.0.1:7996` | captioner |
| `MEMORYSUM_RERANK_BASE_URL` | `http://127.0.0.1:7997` | reranker |
| `MEMORYSUM_ROUTE_POOL` | `120` | candidates per route before fusion |
| `MEMORYSUM_RERANK_TOP_N` | `60` | how many fused candidates are reranked |
| `MEMORYSUM_RETURN_IMAGES` | `true` | inline original images in results |
| `MEMORYSUM_LOG_FILE` | — (stderr) | size-rotated log file |

The service probes all three model servers at startup and logs which ones are
reachable — the first line of the log is the answer to "did my ports line up?".

## 5. Run

**Development / single box (no systemd):**

```bash
./run.sh start      # background uvicorn on :8003, app log in info.log
./run.sh status     # pid + /healthz
./run.sh stop
```

**Directly:**

```bash
.venv/bin/uvicorn memorysum_vl.main:create_app --factory --app-dir src \
    --host 0.0.0.0 --port 8003 --no-access-log
# or, after `pip install .`:
memorysum-vl --host 0.0.0.0 --port 8003
```

**systemd (production-ish):** `deploy/memorysum-vl.service` is a template —
copy it to `/etc/systemd/system/`, adjust `User`/paths, then
`systemctl enable --now memorysum-vl`.

**Docker:**

```bash
docker build -f deploy/Dockerfile -t memorysum-vl:latest .
docker run --rm -p 8003:8003 \
  -v "$PWD/data:/data" \
  -e MEMORYSUM_DB_PATH=/data/memorysum.db \
  -e MEMORYSUM_EMBEDDING_BASE_URL=http://host.docker.internal:7998 \
  -e MEMORYSUM_CAPTION_BASE_URL=http://host.docker.internal:7996 \
  -e MEMORYSUM_RERANK_BASE_URL=http://host.docker.internal:7997 \
  -e MEMORYSUM_AUTH_TOKEN_FILE=/run/secrets/api_token \
  -v "$PWD/.secrets/api_token:/run/secrets/api_token:ro" \
  memorysum-vl:latest
```

`deploy/docker-compose.example.yml` puts the API and all three llama.cpp servers
in one file if you prefer to run everything in containers.

## 6. Tuning

| Symptom | Knob |
|---|---|
| Recall too narrow on long documents | raise `MEMORYSUM_ROUTE_POOL` |
| Latency dominated by reranking | lower `MEMORYSUM_RERANK_TOP_N` / `MEMORYSUM_RERANK_DOC_CHARS` |
| Responses too large / slow to transfer | `MEMORYSUM_RETURN_IMAGES=false`, lower `MEMORYSUM_MAX_RETURNED_IMAGES` |
| Model server queueing under concurrency | `MEMORYSUM_EMBEDDING_CONCURRENCY`, captioner `-np` (one slot per concurrent caption) |
| Log file growing too fast | set `MEMORYSUM_LOG_RESPONSE_BODY_LIMIT` (e.g. `2048`); `0` means **unlimited** |
| Too many irrelevant hits at low confidence | `MEMORYSUM_WEAK_EVIDENCE_THRESHOLD` / `_LIMIT` |
| Answers need adjacent context | `MEMORYSUM_NEIGHBOR_EXPAND` (default 2) |

Rule of thumb: `RERANK_TOP_N ≥ top_k`. The fusion pool is
`max(rerank_top_n, top_k)`, so if you ask for `top_k=100` with
`rerank_top_n=60`, 40 candidates get scored anyway — the knob only decides how
much of the head the cross-encoder sees *before* fusion reordering.

## 7. Operations

```bash
# logs (single file, size-rotated): tail
tail -f info.log

# end-to-end check against a live instance (writes + searches a scratch user)
.venv/bin/python scripts/smoke_api.py

# repair units whose image description is empty, then recompute their vectors
.venv/bin/python -u scripts/backfill_transcripts.py --missing-caption --reembed
```

**Database size.** Everything (text, image bytes, three FTS indexes, two vector
tables) lives in one file. A 500k-memory, 56k-image corpus measured **≈ 30 GB**,
so plan disk accordingly; `data/` is disposable in the sense that it only contains
what you ingest, but it *is* the memory — back it up.

```bash
sqlite3 data/memorysum.db ".backup 'backup.db'"     # consistent, online
```

**Single writer.** SQLite in WAL mode allows one writer at a time. A `Search`
never blocks a write; a long maintenance script does hold the write lock in
batches, so avoid running heavy repairs during peak ingest.

## 8. Troubleshooting

| Log line | Meaning |
|---|---|
| `有模型服务不可达，相关能力会降级` (a model server is unreachable) | the startup probe failed for `embedding`/`captioner`/`reranker` — check the port and `curl /health` |
| `图片向量不可用，视觉召回降级` | embedder rejected the multimodal request (usually the media marker mismatch) |
| `问题实词在语料中无命中，按弃答处理` | abstention fired; the corpus genuinely lacks the question's terms |
| `no such module: vec0` in a script | a bare `sqlite3` connection touched a `vec0` table; load `sqlite_vec` first |
| `unknown model architecture: 'qwen35'` | llama.cpp too old for Rax-4.5 — rebuild |

## 9. Security notes

* The API is **not** multi-tenant-aware beyond `user_id` filtering: anyone with
  the token can read any user's memory by asking for that `user_id`. Treat the
  token as a single-trust-domain credential.
* Bind to `127.0.0.1` unless you terminate TLS in front of it; `run.sh` binds
  `0.0.0.0` for convenience, which is only safe on a private network.
* Image bytes are stored verbatim in the DB and can be returned inline. If you
  ingest sensitive screenshots, remember that `MEMORYSUM_RETURN_IMAGES=true`
  hands them back to any caller holding the token.

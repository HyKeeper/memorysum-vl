# Architecture

`memorysum-vl` is a **memory service** for agents: you `Add` conversations
(text + images), and you `Search` them with questions (text, images, or both).
It is deliberately boring to operate — one Python process, one SQLite file, three
local model servers — while the retrieval path is not boring at all: lexical and
dense routes are fused with RRF and then re-ranked by a cross-encoder.

```
                    ┌──────────────────────────── FastAPI (api/) ───────────────────────────┐
  POST /v1/add      │  routes/add.py   → AddService    ┐                                    │
  POST /v1/search   │  routes/search.py→ SearchService ├── domain/ports.py (protocols)      │
  GET  /healthz     │  security.py (bearer)            ┘            │                          │
                    │  middleware.py (ASGI request/response log)   │                          │
                    └──────────────────────────────────────────────┼──────────────────────────┘
                                                                   │
                                              adapters/stores/sqlite_store.py
                                              (the only implementation today)
                                                    │                    │
                              adapters/stores/repository.py      adapters/llm/*
                              SQLite + FTS5 + sqlite-vec          llama.cpp clients
```

## Layers

| Layer | Path | Responsibility |
|---|---|---|
| **API** | `src/memorysum_vl/api/` | HTTP shape only: routing, bearer auth, ASGI request logging, Pydantic ↔ domain mapping, error envelope |
| **Services** | `src/memorysum_vl/services/` | Orchestration: `AddService.add`, `SearchService.search`. No I/O, no storage details |
| **Domain** | `src/memorysum_vl/domain/` | Pure models (`MemoryUnit`, `MemoryImage`, `Conversation`, `SearchQuery`) and **ports** (`MemoryStore`, `Embedder`, `Captioner`, `Reranker`) |
| **Adapters** | `src/memorysum_vl/adapters/` | Implementations: SQLite store + repository, and the three llama.cpp clients |
| **Core** | `src/memorysum_vl/core/` | Cross-cutting: settings, DI container, logging, errors, RRF fusion, CJK text handling, image preprocessing |

The dependency direction is one-way: `api → services → domain ← adapters`.
`domain/ports.py` is the seam — you can replace SQLite or any model without
touching the API layer. `core/container.py` is the composition root; it is the
only place that knows which concrete adapters are wired together.

## Add: write path

```
messages ──► parse inline images (data URI) ──► sha256 per image
         ──► caption cache lookup (sha256) ──► Rax-4.5 ──► description (+ optional transcript)
         ──► render searchable text per message (role, date, text, caption, transcript)
         ──► unit id = sha1(user_id, session_id, request_id, ordinal)
         ──► SQLite: memory_unit, memory_image, image_caption, fts_*, vec_text, vec_image
         ──► 200 {status: "success", ...}
```

* **A memory unit is one message.** A single `Add` may carry dozens of messages;
  storing the whole conversation as one blob dilutes retrieval, and a benchmark
  question usually maps to one or two sentences.
* **Ids are content-addressed** (`sha1(user, session, request, ordinal)`), so a
  replayed `Add` is *idempotent*: the store deletes the previous rows for that
  `request_id` and rewrites them. Retries upstream cannot duplicate memory.
* **Writes are synchronous.** When `Add` returns `success`, the unit is already
  searchable — captions and vectors included. Nothing is indexed in the
  background, so there is no "eventually consistent" window to reason about.
* **Captions are cached by image sha256** (`image_caption` table). The same
  screenshot across 50 conversations is captioned once, and a query image that
  was already ingested costs no caption call at all.

## Search: retrieval path

```
query text + query images
   │
   ├─ query images ──► caption once (cached by sha256) ──► visual text
   ├─ abstain?  (question content words absent from the corpus → return [])
   │
   ├─ lexical routes (FTS5)          fts_word      : word tokens + camelCase expansions
   │                                 fts_trigram   : character trigrams (typos, partial words)
   │                                 fts_zh        : jieba-cut Chinese + bigrams
   ├─ option route                   options_lexical: every MCQ option is its own query
   ├─ dense routes (sqlite-vec)      dense_text / dense_image for:
   │                                     · the query text vector
   │                                     · each query image vector
   │                                     · a joint text+image vector
   │
   ├─ Reciprocal Rank Fusion (RRF, k adaptive, per-route weights)
   ├─ fetch units → cross-encoder rerank (bge-reranker-v2-m3)
   ├─ fuse rerank ranks with RRF ranks (the prior consensus still counts)
   ├─ weak-evidence clamp · neighbour expansion · session labels · optional images
   └─ top_k
```

Why this order:

| Decision | Reason |
|---|---|
| **RRF before reranking** | Route scores are not comparable (BM25 vs cosine vs logits). RRF only consumes *ranks*, so a weak route cannot dominate just because its scale is larger. Reranking then acts as the final arbiter over the fused head. |
| **`fuse_limit = max(rerank_top_n, top_k)`** | With `top_k=100`, truncating the fused pool at `rerank_top_n=60` silently throws away 40 results before they can be scored. |
| **Query images are captioned too** | A query of "*what is this?*" + image is almost noise to a text cross-encoder: the identical image ranked **2nd** behind a newspaper photo, on logit noise. Captioning the query image (cached) moved it to rank 1 (0.039 → 0.997). |
| **Scores are calibrated, not normalised** | Normalising divides by the max, so the top hit is always `1.0`. Instead, reranker logits go through a sigmoid, so a caller can tell "0.02 = nothing matched" from "0.99 = this is it". |
| **Neighbour expansion** | Multi-hop answers often sit in the adjacent messages of the same session; each hit optionally carries `neighbor_expand` neighbours. |
| **Abstention** | If none of the question's content words appear anywhere in the corpus, returning `[]` beats returning 100 plausible-but-unrelated memories that invite the answer model to hallucinate. |

## Storage model

One SQLite file holds everything: raw text, image bytes, lexical indexes and
vectors. There is no separate vector database to keep in sync.

```sql
memory_unit      (unit_rowid, id UNIQUE, user_id, session_id, request_id, ordinal,
                  role, ts_ms, text, captions, transcripts,
                  lexical, lexical_full, lexical_zh, image_count,
                  caption_state, embedding_state)
memory_image     (image_rowid, memory_id → memory_unit.id, ordinal, media_type,
                  sha256, caption, data BLOB)
image_caption    (sha256 PRIMARY KEY, media_type, model, caption, transcript, created_at)

fts_word         fts5(lexical)      -- word tokens + camelCase expansions
fts_trigram      fts5(lexical)      -- character trigrams
fts_zh           fts5(lexical_zh)   -- Chinese segmentation

vec_text         vec0(unit_rowid  PRIMARY KEY, user_id PARTITION KEY, embedding FLOAT[2048])
vec_image        vec0(image_rowid PRIMARY KEY, user_id PARTITION KEY, embedding FLOAT[2048])
```

* `user_id` is a **partition key** in the `vec0` tables, so KNN search never
  crosses tenants.
* `lexical` is what gets embedded (text + caption, **without** transcripts);
  `lexical_full` (with transcripts) feeds FTS. Transcripts are noisy and help
  exact-term matching far more than they help semantics.
* `caption_state` / `embedding_state` record degradation per unit instead of
  failing the write.

## Degradation

Model servers are treated as *optional*. If one is down:

| Down | Effect | Unit marked |
|---|---|---|
| captioner | images have no description; lexical/dense texts fall back to the message text | `caption_state=missing` |
| embedder | lexical routes only; recall drops but search still answers | `embedding_state=missing` |
| reranker | RRF order is returned as-is | — (logged warning) |

`Add` and `Search` keep returning `200`; only the *quality* of the result changes.
The three servers are probed once at startup and the result is logged, so a
misconfigured port is visible immediately instead of showing up as "search feels
dumb today".

## Operational surface

* **Logging** — one human-readable line per event to a size-rotated file
  (`MEMORYSUM_LOG_FILE`), with request-id correlation injected into every record
  (including service-layer records). Bodies are sanitised: data-URI images are
  collapsed, control characters escaped, and both request and response bodies are
  truncated by `MEMORYSUM_LOG_*_BODY_LIMIT` (0 means "unlimited", which is *not*
  what you want in production — a 100-hit multimodal response can be tens of MiB).
* **Errors** — every failure leaves the API as the same envelope
  (`{"error": {"code", "message"}}`) with a stable `code`; the full reason is
  logged, the caller sees a safe message.
* **Maintenance tools** — `scripts/smoke_api.py` (end-to-end probe against a
  running instance) and `scripts/backfill_transcripts.py` (re-caption units whose
  description is missing/empty and recompute their vectors).

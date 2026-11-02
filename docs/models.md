# Models

`memorysum-vl` uses **three small models, all served locally by
[llama.cpp](https://github.com/ggml-org/llama.cpp)**. Nothing is sent to a
third-party API: text, images and the index all stay on your machine.

All three run comfortably on a single 16 GB consumer GPU (~12.4 GB resident).

| # | Role | Model | Type | Endpoint | GGUF used here |
|---|------|-------|------|----------|----------------|
| 1 | **Image → text** (captioner) | **Rax-4.5** | 1.9B vision-language model (qwen35 architecture) | `:7996` `/v1/chat/completions` | `Rax-4.5.Q8_0.gguf` + `Rax-4.5.mmproj-f16.gguf` |
| 2 | **Embedder** (text + images) | **Qwen3-VL-Embedding-2B** | 2B, 2048-d, one shared vector space for text and images | `:7998` `/v1/embeddings` | `Qwen.Qwen3-VL-Embedding-2B.f16.gguf` + `mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf` |
| 3 | **Reranker** | **bge-reranker-v2-m3** | cross-encoder, multilingual | `:7997` `/rerank` | `bge-reranker-v2-m3-f16.gguf` |

## 1. Rax-4.5 — image description

* Base weights: <https://huggingface.co/raxcore-dev/Rax-4.5>
* GGUF (used here): <https://huggingface.co/mradermacher/Rax-4.5-GGUF>
* Files: `Rax-4.5.Q8_0.gguf` (1.93 GB) + `Rax-4.5.mmproj-f16.gguf` (0.62 GB)
* Served with: `deploy/llama/start-captioner.sh`
* Endpoint used: `POST /v1/chat/completions` (OpenAI-compatible)

Rax-4.5 is a **reasoning** model: by default the answer lands in
`reasoning_content` and `content` comes back empty. The client must send
`chat_template_kwargs: {"enable_thinking": false}`; the adapter does this for you
(`src/memorysum_vl/adapters/llm/caption.py`). With thinking disabled it returns a
description in ~0.7–1.7 s per image.

The captioner is *optional at runtime*: if it is down, `Add` still succeeds and
the affected units are marked `caption_state=missing` (see
[Degradation](architecture.md#degradation)).

## 2. Qwen3-VL-Embedding-2B — text and image embeddings

* Base weights: <https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B>
* GGUF (used here): <https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF>
* Files: `Qwen.Qwen3-VL-Embedding-2B.f16.gguf` (3.21 GB) +
  `mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf` (0.76 GB)
* Served with: `deploy/llama/start-embedder.sh`
* Endpoint used: `POST /v1/embeddings`
* Dimension: **2048** — must match `MEMORYSUM_EMBEDDING_DIM`, because the
  dimension is baked into the `sqlite-vec` virtual tables at first start.

This model is the reason the project can do image search without a separate
image encoder: a photo and its description land close together in the same
space, so `dense_text` and `dense_image` routes are directly comparable.

## 3. bge-reranker-v2-m3 — cross-encoder reranking

* Base weights: <https://huggingface.co/BAAI/bge-reranker-v2-m3>
* GGUF: <https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF> (publishes `Q8_0`;
  this deployment used an **f16** conversion, 1.08 GB — any llama.cpp-compatible
  `bge-reranker-v2-m3` GGUF works)
* Served with: `deploy/llama/start-reranker.sh`
* Endpoint used: `POST /rerank` (`--reranking --pooling rank`)

## Runtime: llama.cpp

* Repository: <https://github.com/ggml-org/llama.cpp>
* Build with CUDA: `cmake -B build -DGGML_CUDA=ON && cmake --build build -j`
* Needs a version recent enough to support the `qwen35` architecture (Rax-4.5)
  and multimodal embedding (`--embeddings` + `--mmproj`). If `llama-server`
  refuses the model with `unknown model architecture: 'qwen35'`, rebuild from a
  newer commit.

## Downloading the weights

```bash
# pick a directory; the start scripts read MODEL_DIR (default: ~/models)
export MODEL_DIR=~/models
mkdir -p "$MODEL_DIR"

# Hugging Face CLI (pip install -U huggingface_hub)
hf download mradermacher/Rax-4.5-GGUF \
  Rax-4.5.Q8_0.gguf Rax-4.5.mmproj-f16.gguf --local-dir "$MODEL_DIR"

hf download DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF \
  Qwen.Qwen3-VL-Embedding-2B.f16.gguf \
  mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf --local-dir "$MODEL_DIR"

hf download gpustack/bge-reranker-v2-m3-GGUF \
  bge-reranker-v2-m3-Q8_0.gguf --local-dir "$MODEL_DIR"
```

If `huggingface.co` is unreachable from your network, set a mirror — the CLI and
`curl` both honour `HF_ENDPOINT`:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

`deploy/llama/download-models.sh` wraps the three commands above.

## Verifying what you downloaded

| File | Expected size |
|------|---------------|
| `Rax-4.5.Q8_0.gguf` | ≈ 1.93 GB |
| `Rax-4.5.mmproj-f16.gguf` | 819,395,136 B |
| `Qwen.Qwen3-VL-Embedding-2B.f16.gguf` | ≈ 3.21 GB |
| `mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf` | ≈ 0.76 GB |
| reranker GGUF | ≈ 1.08 GB (f16) / ≈ 0.6 GB (Q8_0) |

## Model gotchas (all of these were real failures)

| Symptom | Cause | Fix |
|---|---|---|
| `clip_init: failed to load model ... failed to seek for tensor v.blk.0...` | the mmproj (vision tower) is a **separate** file and was missing/truncated | download the `mmproj-*` file next to the main GGUF |
| `number of media markers in text (0) does not match number of bitmaps (1)` | llama.cpp **randomises** the multimodal placeholder per process | pin it: `LLAMA_MEDIA_MARKER='<__media__>'` on the server and `MEMORYSUM_EMBEDDING_MEDIA_MARKER='<__media__>'` on the client |
| `failed to find a memory slot for batch of size 2023` | a large image can produce 2000+ vision tokens and overflow the ubatch | `--image-max-tokens 1536` on the server, `MEMORYSUM_IMAGE_MAX_SIDE=1280` on the client |
| captioning returns an empty string | Rax-4.5 is a reasoning model and spent `max_tokens` on `reasoning_content` | `chat_template_kwargs.enable_thinking=false` |
| image embeddings are slow/over budget | f16 on a small GPU | captioner at `Q8_0`, embedder is the heavy one — give it `-np 8` and `-c 16384` |

## Swapping a model

* **Text-only deployment?** Any `Qwen3-Embedding-*` GGUF works with
  `MEMORYSUM_EMBEDDING_*`; images then simply have no vector route
  (`dense_image` returns nothing) and `MEMORYSUM_CAPTION_*` can point at any
  chat model that accepts images.
* **Different reranker?** The adapter speaks the llama.cpp `/rerank` contract
  (`TEI`-compatible). Point `MEMORYSUM_RERANK_BASE_URL` at it; set
  `MEMORYSUM_RERANK_ENABLED=false` to skip reranking entirely.
* Changing the **embedding dimension** requires a fresh database
  (`data/` is a local cache of your own memory; nothing else depends on it).

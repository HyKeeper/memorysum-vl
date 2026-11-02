#!/usr/bin/env bash
# Start the embedder (Qwen3-VL-Embedding-2B) on :7998 — llama.cpp /v1/embeddings.
#
#   MODEL_DIR=~/models ./deploy/llama/start-embedder.sh
#
# Text and images land in the same 2048-d space, which is what lets the store
# run a text vector against image vectors in one query.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7998}"
NGL="${NGL:-99}"
PARALLEL="${PARALLEL:-8}"     # slots; the client's EMBEDDING_CONCURRENCY should fit in this

# --image-max-tokens caps one image at 1536 vision tokens: a 1280px image would
# otherwise produce 2000+, overflow the ubatch, and fail with
# "failed to find a memory slot for batch of size ...".
exec env LLAMA_MEDIA_MARKER='<__media__>' "$LLAMA_SERVER" \
  -m "$MODEL_DIR/Qwen.Qwen3-VL-Embedding-2B.f16.gguf" \
  --mmproj "$MODEL_DIR/mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf" \
  --embeddings --pooling last -fa on \
  -c 16384 -b 4096 -ub 2048 -np "$PARALLEL" \
  --image-min-tokens 256 --image-max-tokens 1536 \
  -ngl "$NGL" --cache-ram 0 \
  --host "$HOST" --port "$PORT" --alias qwen3-vl-embedding-2b

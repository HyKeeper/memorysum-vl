#!/usr/bin/env bash
# Start the reranker (bge-reranker-v2-m3) on :7997 — llama.cpp /rerank.
#
#   MODEL_DIR=~/models ./deploy/llama/start-reranker.sh
#
# Set RERANK_MODEL_FILE if you downloaded a differently named quant.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7997}"
NGL="${NGL:-99}"
RERANK_MODEL_FILE="${RERANK_MODEL_FILE:-bge-reranker-v2-m3-Q8_0.gguf}"

exec "$LLAMA_SERVER" \
  -m "$MODEL_DIR/$RERANK_MODEL_FILE" \
  --reranking --pooling rank -fa on \
  -c 8192 -b 8192 -ub 8192 -ngl "$NGL" --cache-ram 0 \
  --host "$HOST" --port "$PORT" --alias bge-reranker-v2-m3

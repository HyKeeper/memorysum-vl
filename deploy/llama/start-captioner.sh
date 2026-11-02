#!/usr/bin/env bash
# Start the image captioner (Rax-4.5) on :7996 — llama.cpp, OpenAI-compatible.
#
#   MODEL_DIR=~/models ./deploy/llama/start-captioner.sh
#
# Environment:
#   MODEL_DIR       directory holding the GGUFs (default: ~/models)
#   LLAMA_SERVER    llama-server binary (default: found on PATH)
#   HOST / PORT     bind address (default: 127.0.0.1:7996)
#   NGL             layers offloaded to GPU (default: 99; use 0 for CPU)
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7996}"
NGL="${NGL:-99}"

# Why LLAMA_MEDIA_MARKER: llama.cpp randomises the multimodal placeholder per
# process; the client cannot guess it. Pinning it here keeps the two sides in
# sync with MEMORYSUM_EMBEDDING_MEDIA_MARKER.
exec env LLAMA_MEDIA_MARKER='<__media__>' "$LLAMA_SERVER" \
  -m "$MODEL_DIR/Rax-4.5.Q8_0.gguf" \
  --mmproj "$MODEL_DIR/Rax-4.5.mmproj-f16.gguf" \
  --jinja -fa on -c 8192 -b 2048 -ub 1024 -ngl "$NGL" --cache-ram 0 \
  --host "$HOST" --port "$PORT" --alias rax-4.5

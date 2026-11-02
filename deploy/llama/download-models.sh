#!/usr/bin/env bash
# Download the three GGUFs used by memorysum-vl into $MODEL_DIR.
#
#   MODEL_DIR=~/models ./deploy/llama/download-models.sh
#
# Uses the Hugging Face CLI when available, `curl` otherwise. If huggingface.co
# is blocked in your network, set HF_ENDPOINT — both tools honour it:
#
#   HF_ENDPOINT=https://hf-mirror.com ./deploy/llama/download-models.sh
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_ENDPOINT="${HF_ENDPOINT%/}"
mkdir -p "$MODEL_DIR"

files=(
  "mradermacher/Rax-4.5-GGUF|Rax-4.5.Q8_0.gguf"
  "mradermacher/Rax-4.5-GGUF|Rax-4.5.mmproj-f16.gguf"
  "DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF|Qwen.Qwen3-VL-Embedding-2B.f16.gguf"
  "DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF|mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf"
  "gpustack/bge-reranker-v2-m3-GGUF|bge-reranker-v2-m3-Q8_0.gguf"
)

have_cli=0
command -v hf >/dev/null 2>&1 && have_cli=1
command -v huggingface-cli >/dev/null 2>&1 && have_cli=1

for entry in "${files[@]}"; do
  repo="${entry%%|*}"
  name="${entry##*|}"
  if [ -s "$MODEL_DIR/$name" ]; then
    echo "skip  $name (already present)"
    continue
  fi
  echo "fetch $repo / $name"
  if [ "$have_cli" = 1 ]; then
    HF_ENDPOINT="$HF_ENDPOINT" hf download "$repo" "$name" --local-dir "$MODEL_DIR" 2>/dev/null \
      || HF_ENDPOINT="$HF_ENDPOINT" huggingface-cli download "$repo" "$name" --local-dir "$MODEL_DIR"
  else
    # -C - resumes an interrupted download instead of restarting a multi-GB file
    curl -fL -C - --retry 5 --retry-delay 3 \
      -o "$MODEL_DIR/$name" \
      "$HF_ENDPOINT/$repo/resolve/main/$name?download=true"
  fi
done

echo
echo "models in $MODEL_DIR:"
ls -lh "$MODEL_DIR"/*.gguf 2>/dev/null || true
echo
echo "expected: Rax-4.5.Q8_0 ≈1.93G · Rax-4.5.mmproj-f16 =819395136B ·"
echo "          Qwen3-VL-Embedding-2B.f16 ≈3.21G · mmproj ≈0.76G · reranker ≈0.6G"
echo "a much smaller file means the download was truncated — re-run this script"

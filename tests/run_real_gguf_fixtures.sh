#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 tiny-llm.gguf tinyllama-stories.gguf" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ZEN_BUILD_DIR:-$ROOT/build}"
for model in "$1" "$2"; do
  echo "=== inspect: $model ==="
  "$BUILD/zen_model_inspect" "$model"
  echo "=== decode all tensors: $model ==="
  "$BUILD/zen_tensor_check" "$model"
  echo "=== text generation: $model ==="
  "$BUILD/zen_generate_ids" "$model" --prompt-text "Once upon a time" --max-new 8
done

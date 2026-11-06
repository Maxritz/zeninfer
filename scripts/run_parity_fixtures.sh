#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: $0 <zeninfer-build-dir> <tiny-llm.gguf> <tinyllama-stories.gguf>" >&2
  exit 2
fi
B=$(cd "$1" && pwd)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${ZEN_PARITY_OUT:-"$ROOT/parity-results"}
mkdir -p "$OUT"
run_one() {
  local model="$1" tag="$2" index=0
  "$B/zen_parity_export" "$model" "$OUT/$tag.znp"
  for item in 'Once upon a time' 'The quick brown fox jumps over the lazy dog.' 'Hello, world!' $'Once upon a time\nthere was a little girl.'; do
    index=$((index + 1))
    "$B/zen_parity_probe" "$model" --prompt-text "$item" "$OUT/$tag.$index.logits" > "$OUT/$tag.$index.prompt"
    local ids
    ids=$(sed -n 's/^prompt_ids=//p' "$OUT/$tag.$index.prompt")
    python3 "$ROOT/scripts/numpy_reference.py" "$OUT/$tag.znp" "$OUT/$tag.$index.logits" \
      --prompt-ids "$ids" --json-out "$OUT/$tag.$index.json"
  done
}
run_one "$2" tiny-llm
run_one "$3" tinyllama-stories
echo "Parity results: $OUT"

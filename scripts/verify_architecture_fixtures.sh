#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${ROOT}/build-avx2}"
if [[ ! -x "${BUILD_DIR}/zen_arch_microbench" ]]; then
  echo "Missing zen_arch_microbench in ${BUILD_DIR}; build the project first" >&2
  exit 2
fi

MODELS=(
  "${ROOT}/models/synthetic/qwen3-moe-tiny.zenmodel"
  "${ROOT}/models/synthetic/bitnet-tiny.zenmodel"
  "${ROOT}/models/synthetic/bonsai-ternary-tiny.zenmodel"
)

"${BUILD_DIR}/zen_arch_microbench" "${MODELS[@]}"
ctest --test-dir "${BUILD_DIR}" --output-on-failure -R 'zen_arch_micro_models|zen_core_tests'

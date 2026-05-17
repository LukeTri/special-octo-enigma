#!/usr/bin/env bash
set -euo pipefail

# Runs the Parameter Golf language-model objective with one or more attention modes.
#
# Common overrides:
#   MODES="baseline distance_prefix" ITERATIONS=2000 MAX_WALLCLOCK_SECONDS=0 bash scripts/run_parameter_golf_attention.sh
#   DATA_PATH=/workspace/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_1024_bpe.model bash scripts/run_parameter_golf_attention.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODES="${MODES:-baseline distance_prefix distance_per_band}"
ATTENTION_BANDS="${ATTENTION_BANDS:-128:64,512:32,inf:16}"
ATTENTION_VALUE_BANDS="${ATTENTION_VALUE_BANDS:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-dense}"
ATTENTION_CHUNK_SIZE="${ATTENTION_CHUNK_SIZE:-128}"
ATTENTION_FLEX_BLOCK_SIZE="${ATTENTION_FLEX_BLOCK_SIZE:-128}"
ATTENTION_CHECKPOINT="${ATTENTION_CHECKPOINT:-1}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/parameter_golf/$(date +%Y%m%d_%H%M%S)}"

DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"

mkdir -p "${RUN_ROOT}"

echo "[pg] repo: ${REPO_ROOT}"
echo "[pg] run root: ${RUN_ROOT}"
echo "[pg] modes: ${MODES}"
echo "[pg] bands: ${ATTENTION_BANDS}"
if [[ -n "${ATTENTION_VALUE_BANDS}" ]]; then
    echo "[pg] value bands: ${ATTENTION_VALUE_BANDS}"
fi
echo "[pg] attention backend: ${ATTENTION_BACKEND}"
echo "[pg] attention chunk size: ${ATTENTION_CHUNK_SIZE}"
echo "[pg] flex block size: ${ATTENTION_FLEX_BLOCK_SIZE}"
echo "[pg] attention checkpoint: ${ATTENTION_CHECKPOINT}"
echo "[pg] data: ${DATA_PATH}"
echo "[pg] tokenizer: ${TOKENIZER_PATH}"

for mode in ${MODES}; do
  run_dir="${RUN_ROOT}/${mode}"
  mkdir -p "${run_dir}"
  echo
  echo "[pg] ===== ${mode} ====="
  (
    cd "${run_dir}"
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    DATA_PATH="${DATA_PATH}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    ATTENTION_MODE="${mode}" \
    ATTENTION_BANDS="${ATTENTION_BANDS}" \
    ATTENTION_VALUE_BANDS="${ATTENTION_VALUE_BANDS}" \
    ATTENTION_BACKEND="${ATTENTION_BACKEND}" \
    ATTENTION_CHUNK_SIZE="${ATTENTION_CHUNK_SIZE}" \
    ATTENTION_FLEX_BLOCK_SIZE="${ATTENTION_FLEX_BLOCK_SIZE}" \
    ATTENTION_CHECKPOINT="${ATTENTION_CHECKPOINT}" \
    RUN_ID="${RUN_ID:-${mode}}" \
    python "${REPO_ROOT}/parameter_golf_distance_attention.py"
  )
done

echo
echo "[pg] complete: ${RUN_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

# Downloads the cached FineWeb data used by OpenAI's Parameter Golf baseline.
# Default TRAIN_SHARDS=32 is a practical pilot size; use TRAIN_SHARDS=300 for the
# upstream default training split size.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/openai/parameter-golf.git}"
UPSTREAM_DIR="${UPSTREAM_DIR:-${REPO_ROOT}/.upstream/parameter-golf}"
VARIANT="${VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-32}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-1}"

DATASET_NAME="fineweb10B_${VARIANT}"
TOKENIZER_NAME="fineweb_1024_bpe.model"
UPSTREAM_DATASET="${UPSTREAM_DIR}/data/datasets/${DATASET_NAME}"
UPSTREAM_TOKENIZER="${UPSTREAM_DIR}/data/tokenizers/${TOKENIZER_NAME}"
LOCAL_DATASET="${REPO_ROOT}/data/datasets/${DATASET_NAME}"
LOCAL_TOKENIZER="${REPO_ROOT}/data/tokenizers/${TOKENIZER_NAME}"

link_or_keep_existing() {
  local source_path="$1"
  local link_path="$2"

  mkdir -p "$(dirname "${link_path}")"

  if [ -L "${link_path}" ]; then
    ln -sfn "${source_path}" "${link_path}"
    return 0
  fi

  if [ -e "${link_path}" ]; then
    echo "[data] Keeping existing non-symlink path: ${link_path}"
    echo "[data] If this is stale, move it aside and rerun this script."
    return 0
  fi

  ln -s "${source_path}" "${link_path}"
}

command -v git >/dev/null 2>&1 || { echo "[data] git is required" >&2; exit 1; }
command -v python >/dev/null 2>&1 || { echo "[data] python is required" >&2; exit 1; }

mkdir -p "$(dirname "${UPSTREAM_DIR}")"

if [ -d "${UPSTREAM_DIR}/.git" ]; then
  echo "[data] Updating upstream Parameter Golf repo: ${UPSTREAM_DIR}"
  git -C "${UPSTREAM_DIR}" pull --ff-only
else
  echo "[data] Cloning upstream Parameter Golf repo: ${UPSTREAM_DIR}"
  git clone "${UPSTREAM_URL}" "${UPSTREAM_DIR}"
fi

if [ "${INSTALL_REQUIREMENTS}" = "1" ]; then
  echo "[data] Installing local Python requirements"
  python -m pip install -r "${REPO_ROOT}/requirements.txt"
else
  echo "[data] Skipping requirement install because INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS}"
fi

echo "[data] Downloading cached FineWeb data"
echo "[data] variant=${VARIANT} train_shards=${TRAIN_SHARDS}"
(
  cd "${UPSTREAM_DIR}"
  python data/cached_challenge_fineweb.py --variant "${VARIANT}" --train-shards "${TRAIN_SHARDS}"
)

if [ ! -d "${UPSTREAM_DATASET}" ]; then
  echo "[data] Expected dataset was not created: ${UPSTREAM_DATASET}" >&2
  exit 1
fi

if [ ! -f "${UPSTREAM_TOKENIZER}" ]; then
  echo "[data] Expected tokenizer was not created: ${UPSTREAM_TOKENIZER}" >&2
  exit 1
fi

link_or_keep_existing "${UPSTREAM_DATASET}" "${LOCAL_DATASET}"
link_or_keep_existing "${UPSTREAM_TOKENIZER}" "${LOCAL_TOKENIZER}"

echo
echo "[data] Ready for Parameter Golf runs:"
echo "[data] DATA_PATH=${LOCAL_DATASET}"
echo "[data] TOKENIZER_PATH=${LOCAL_TOKENIZER}"

echo
echo "[data] Size summary:"
du -sh "${UPSTREAM_DATASET}" "${UPSTREAM_TOKENIZER}" 2>/dev/null || true

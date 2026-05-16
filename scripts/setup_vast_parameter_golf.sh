#!/usr/bin/env bash
set -euo pipefail

# One-shot Vast.ai setup for the Parameter Golf attention experiments.
# Run from a freshly pulled repo:
#   bash scripts/setup_vast_parameter_golf.sh
#
# Common overrides:
#   TRAIN_SHARDS=300 bash scripts/setup_vast_parameter_golf.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash scripts/setup_vast_parameter_golf.sh
#   SKIP_DATA=1 bash scripts/setup_vast_parameter_golf.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TRAIN_SHARDS="${TRAIN_SHARDS:-32}"
SKIP_DATA="${SKIP_DATA:-0}"

cd "${REPO_ROOT}"

echo "[setup] repo: ${REPO_ROOT}"
echo "[setup] venv: ${VENV_DIR}"
echo "[setup] torch index: ${TORCH_INDEX_URL}"
echo "[setup] train shards: ${TRAIN_SHARDS}"

if ! command -v git >/dev/null 2>&1; then
  echo "[setup] git is required" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[setup] ${PYTHON_BIN} is required" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "[setup] GPU info:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  echo "[setup] Warning: nvidia-smi was not found. This setup expects a CUDA GPU on Vast." >&2
fi

echo
echo "[setup] Creating/updating virtualenv"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install PyTorch separately so Vast machines with CUDA 12.4-era drivers do not
# accidentally get a newer default wheel that cannot initialize CUDA.
echo
echo "[setup] Installing PyTorch from ${TORCH_INDEX_URL}"
python -m pip uninstall -y torch >/dev/null 2>&1 || true
python -m pip install torch --index-url "${TORCH_INDEX_URL}"

echo
echo "[setup] Installing non-Torch requirements"
REQ_TMP="$(mktemp)"
grep -Ev '^[[:space:]]*torch([<>=~! ]|$)' requirements.txt > "${REQ_TMP}"
python -m pip install -r "${REQ_TMP}"
rm -f "${REQ_TMP}"

echo
echo "[setup] Checking CUDA from PyTorch"
python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA. Try a different TORCH_INDEX_URL or Vast image.")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

if [ "${SKIP_DATA}" != "1" ]; then
  echo
  echo "[setup] Downloading/linking Parameter Golf data"
  INSTALL_REQUIREMENTS=0 TRAIN_SHARDS="${TRAIN_SHARDS}" bash scripts/download_parameter_golf_data.sh
else
  echo
  echo "[setup] Skipping data download because SKIP_DATA=1"
fi

echo
echo "[setup] Verifying expected data paths"
if [ -f data/tokenizers/fineweb_1024_bpe.model ] && [ -d data/datasets/fineweb10B_sp1024 ]; then
  ls -lh data/tokenizers/fineweb_1024_bpe.model
  find data/datasets/fineweb10B_sp1024 -maxdepth 1 -name 'fineweb_train_*.bin' | head -5
elif [ "${SKIP_DATA}" = "1" ]; then
  echo "[setup] Data paths are missing, but SKIP_DATA=1 was set."
else
  echo "[setup] Expected Parameter Golf data paths are missing." >&2
  exit 1
fi

echo
echo "[setup] Ready. Activate with:"
echo "source ${VENV_DIR}/bin/activate"
echo
echo "[setup] Suggested pilot run:"
echo 'ATTENTION_BANDS="128:64,512:32,inf:16" bash scripts/run_parameter_golf_attention.sh'

#!/usr/bin/env bash
set -euo pipefail

# Required env:
#   REPO_URL="https://github.com/<user>/<repo>.git"
#
# Optional env:
#   REPO_DIR="/workspace/efficient-attention"
#   REPO_BRANCH="main"
#   PYTHON_BIN="python3"
#   RUN_ARGS="--mode compare --device cuda --steps 2000"
#
# Example:
#   REPO_URL="https://github.com/you/efficient-attention.git" \
#   RUN_ARGS="--mode compare --device cuda --steps 2000 --bands 64:16,256:8,inf:4" \
#   bash vast_onstart.sh

if [[ -z "${REPO_URL:-}" ]]; then
  echo "REPO_URL is required. Example: https://github.com/<user>/<repo>.git"
  exit 1
fi

REPO_DIR="${REPO_DIR:-/workspace/efficient-attention}"
REPO_BRANCH="${REPO_BRANCH:-main}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ARGS="${RUN_ARGS:---mode compare --device cuda --steps 2000 --log-interval 100}"

mkdir -p /workspace

if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "[onstart] Repo exists, updating..."
  git -C "${REPO_DIR}" fetch --all
  git -C "${REPO_DIR}" checkout "${REPO_BRANCH}"
  git -C "${REPO_DIR}" pull --ff-only
else
  echo "[onstart] Cloning repo..."
  git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

echo "[onstart] Creating virtualenv..."
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate

echo "[onstart] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p runs
LOG_FILE="runs/vast_run_$(date +%Y%m%d_%H%M%S).log"
echo "[onstart] Running experiment..."
echo "[onstart] Command: python distance_band_experiment.py ${RUN_ARGS}"
python distance_band_experiment.py ${RUN_ARGS} | tee "${LOG_FILE}"

echo "[onstart] Done. Log written to ${LOG_FILE}"


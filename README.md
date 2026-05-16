# Distance-Banded Attention Prototype

Prototype for testing a modified attention score:

`score(i, j) = (P_{i-j} q_i)^T (P_{i-j} k_j)`

where `P_{i-j}` is distance-conditioned through causal bands and reduced per-band dimensions.

## Implemented Models

- `baseline`: standard causal attention (`q_i^T k_j`)
- `distance_prefix`: shared Q/K, per-band prefix truncation
- `distance_per_band`: separate learned Q/K projections per band

## Benchmark

`distance_band_experiment.py` runs a synthetic long-range associative recall task and reports:

- overall answer accuracy
- distance-binned accuracy
- result JSON saved to `runs/<run_name>.json`

By default, each sequence uses unique key tokens and trains on every key->value pair plus the final query. This avoids ambiguous repeated-key examples and gives the model dense in-context learning signal.

---

## Local Quickstart (Mac/Linux)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python distance_band_experiment.py \
  --mode compare \
  --bands "64:16,256:8,inf:4" \
  --d-model 128 --n-heads 4 --n-layers 4 \
  --seq-len 256 --num-pairs 96 --key-vocab 128 \
  --steps 600
```

### Useful Speed Flags (GPU)

- `--device cuda`
- `--precision auto` (chooses bf16/fp16 on CUDA)
- `--torch-compile` (PyTorch 2.x compile path)

Example:

```bash
python distance_band_experiment.py \
  --mode compare \
  --device cuda \
  --precision auto \
  --torch-compile \
  --steps 2000 \
  --log-interval 100
```

### Robust Evaluation Protocol (Recommended)

Single runs are noisy. Use multi-seed paired comparisons:

- run each mode with the same `--seed` and data seeds derived from it
- aggregate over multiple seeds
- compare deltas vs baseline with confidence intervals and sign test

Run robust sweep:

```bash
python scripts/robust_experiment.py \
  --seeds "11,22,33,44,55,66,77" \
  --modes "baseline,distance_prefix,distance_per_band" \
  --metric answer_acc \
  --extra-args "--device cuda --precision auto --torch-compile --steps 2000 --log-interval 100 --num-pairs 96 --key-vocab 128 --bands 64:16,256:8,inf:4"
```

Outputs:

- raw per-run JSONs: `runs/robust/raw/`
- aggregate summary: `runs/robust/summary.json`
- readable report: `runs/robust/summary.md`
- includes runtime + throughput stats (`runtime_seconds`, `train_tokens_per_sec`) and paired speedup-vs-baseline analysis

---

## Put This On GitHub

From this project directory:

```bash
git init
git add .
git commit -m "Initial distance-banded attention prototype"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

`.gitignore` excludes `.venv/`, `__pycache__/`, and `runs/`.

---

## Run On Vast.ai (SSH/Jupyter launch mode)

### Option A: Paste On-Start Script In Vast UI

Paste this in the Vast.ai **On-start script** field (edit repo URL/branch/args):

```bash
set -euo pipefail
cd /workspace
git clone --branch main https://github.com/<your-user>/<your-repo>.git efficient-attention
cd efficient-attention
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python distance_band_experiment.py \
  --mode compare \
  --device cuda \
  --precision auto \
  --torch-compile \
  --steps 2000 \
  --log-interval 100
```

### Option B: Use Included Script

After cloning once on the instance:

```bash
cd /workspace/efficient-attention
REPO_URL="https://github.com/<your-user>/<your-repo>.git" \
REPO_BRANCH="main" \
RUN_ARGS="--mode compare --device cuda --precision auto --torch-compile --steps 2000 --log-interval 100" \
bash scripts/vast_onstart.sh
```

The script updates the repo, installs deps, runs training, and writes logs into `runs/`.

---

## Core CLI Flags

- `--mode`: `compare|baseline|distance_prefix|distance_per_band`
- `--bands`: e.g. `64:16,256:8,inf:4` (dims are per head)
- `--key-vocab`: must be at least `--num-pairs` unless `--allow-key-repeats` is set
- `--target-mode`: `all_values` (default) or `final_only`
- `--device`: `auto|cpu|cuda|mps`
- `--precision`: `auto|fp32|bf16|fp16`
- `--torch-compile`
- `--out-dir` and `--run-name` for results JSON organization

---

## Parameter Golf LM Objective

`parameter_golf_distance_attention.py` adapts the Parameter Golf training script from the local KFAC/Muon setup and swaps in configurable attention modes while keeping the real FineWeb token-LM objective, validation loss, BPB metric, Muon optimizer, and throughput logging.

Modes are selected with environment variables:

- `ATTENTION_MODE=baseline`
- `ATTENTION_MODE=distance_prefix`
- `ATTENTION_MODE=distance_per_band`
- `ATTENTION_BANDS=128:64,512:32,inf:16`

Run a quick single mode:

```bash
ATTENTION_MODE=distance_prefix \
ATTENTION_BANDS="128:64,512:32,inf:16" \
ITERATIONS=2000 \
MAX_WALLCLOCK_SECONDS=0 \
python parameter_golf_distance_attention.py
```

Run all modes into separate output folders:

```bash
ITERATIONS=2000 \
MAX_WALLCLOCK_SECONDS=0 \
ATTENTION_BANDS="128:64,512:32,inf:16" \
bash scripts/run_parameter_golf_attention.sh
```

The script expects Parameter Golf data at:

- `data/datasets/fineweb10B_sp1024`
- `data/tokenizers/fineweb_1024_bpe.model`

Override with:

```bash
DATA_PATH=/workspace/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_1024_bpe.model \
bash scripts/run_parameter_golf_attention.sh
```

Note: the current distance-banded implementation is a correctness/quality prototype and still uses dense score tensors internally. It is useful for LM-quality comparisons first; a real speed win needs a more fused/windowed implementation.

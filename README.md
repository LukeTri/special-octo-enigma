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

Fresh Vast setup after cloning/pulling:

```bash
bash scripts/setup_vast_parameter_golf.sh
```

This creates `.venv`, installs dependencies, downloads/links the cached Parameter Golf FineWeb data, and checks that CUDA plus the tokenizer/data paths work. By default it uses the normal PyPI/default PyTorch package; set `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126` only if you need to force a specific CUDA wheel.

Modes are selected with environment variables:

- `ATTENTION_MODE=baseline`
- `ATTENTION_MODE=distance_prefix`
- `ATTENTION_MODE=distance_per_band`
- `ATTENTION_BANDS=128:64,512:32,inf:16`
- `ATTENTION_CHUNK_SIZE=128`
- `ATTENTION_CHECKPOINT=1`

The custom distance modes use chunked/checkpointed attention by default to avoid storing full dense `seq_len x seq_len` score tensors for every band during backward.

Run a quick single mode:

```bash
ATTENTION_MODE=distance_prefix \
ATTENTION_BANDS="128:64,512:32,inf:16" \
ATTENTION_CHUNK_SIZE=128 \
ITERATIONS=2000 \
MAX_WALLCLOCK_SECONDS=0 \
python parameter_golf_distance_attention.py
```

Run all modes into separate output folders:

```bash
ITERATIONS=2000 \
MAX_WALLCLOCK_SECONDS=0 \
ATTENTION_BANDS="128:64,512:32,inf:16" \
ATTENTION_CHUNK_SIZE=128 \
bash scripts/run_parameter_golf_attention.sh
```

The script expects Parameter Golf data at:

- `data/datasets/fineweb10B_sp1024`
- `data/tokenizers/fineweb_1024_bpe.model`

Download/link the official cached Parameter Golf FineWeb data:

```bash
TRAIN_SHARDS=32 bash scripts/download_parameter_golf_data.sh
```

`TRAIN_SHARDS=32` is a good pilot size for debugging. Use `TRAIN_SHARDS=300` for the upstream default training split size.

Override with:

```bash
DATA_PATH=/workspace/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_1024_bpe.model \
bash scripts/run_parameter_golf_attention.sh
```

Note: the current distance-banded implementation is a correctness/quality prototype. It now chunks and checkpoints the dense score computation to avoid the earlier OOM path, but a real speed win still needs a more fused/windowed implementation.

#!/usr/bin/env python3
"""
Run a robust multi-seed comparison across attention variants and summarize noise.

This script repeatedly calls `distance_band_experiment.py` with different seeds,
aggregates metrics, computes bootstrap confidence intervals, and reports paired
comparisons against baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class MetricStats:
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float
    min_val: float
    max_val: float


def parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    if not out:
        raise ValueError("Expected at least one integer in list.")
    return out


def parse_str_list(spec: str) -> List[str]:
    out = [x.strip() for x in spec.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one entry.")
    return out


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    if n == 1:
        return [float(arr[0]), float(arr[0])]
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    means = arr[indices].mean(axis=1)
    q = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return [float(q[0]), float(q[1])]


def summarize_metric(
    values: Sequence[float],
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> MetricStats:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.shape[0])
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    ci_low, ci_high = bootstrap_ci(arr.tolist(), n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
    return MetricStats(
        n=n,
        mean=mean,
        std=std,
        ci_low=ci_low,
        ci_high=ci_high,
        min_val=float(arr.min()),
        max_val=float(arr.max()),
    )


def sign_test_two_sided(deltas: Sequence[float]) -> Dict[str, float]:
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    n = wins + losses
    if n == 0:
        return {
            "wins": float(wins),
            "losses": float(losses),
            "ties": float(ties),
            "n_effective": float(n),
            "win_rate": 0.0,
            "p_value": 1.0,
        }
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p_value = min(1.0, 2.0 * tail)
    return {
        "wins": float(wins),
        "losses": float(losses),
        "ties": float(ties),
        "n_effective": float(n),
        "win_rate": float(wins / n),
        "p_value": float(p_value),
    }


def run_once(
    python_bin: str,
    experiment_script: Path,
    mode: str,
    seed: int,
    out_dir: Path,
    extra_args: Sequence[str],
) -> Tuple[Path, float]:
    run_name = f"robust_s{seed}_{mode}"
    cmd = [
        python_bin,
        str(experiment_script),
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        "--out-dir",
        str(out_dir),
    ]
    cmd.extend(extra_args)
    print(f"\n[run] mode={mode} seed={seed}", flush=True)
    print(f"[cmd] {' '.join(shlex.quote(x) for x in cmd)}", flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    runtime_seconds = time.perf_counter() - t0
    out_path = out_dir / f"{run_name}.json"
    if not out_path.exists():
        raise FileNotFoundError(f"Expected output JSON not found: {out_path}")
    return out_path, runtime_seconds


def load_run_json(path: Path, mode: str) -> Dict[str, float]:
    payload = json.loads(path.read_text())
    results = payload.get("results", {})
    if mode not in results:
        # In case future code writes only one key with a different alias.
        if len(results) == 1:
            return next(iter(results.values()))
        raise KeyError(f"Mode '{mode}' missing in {path}. Keys: {list(results.keys())}")
    return results[mode]


def infer_train_tokens_seen(run_payload: Dict[str, object]) -> float:
    args = run_payload.get("args", {})
    if not isinstance(args, dict):
        return 0.0
    steps = int(args.get("steps", 0) or 0)
    batch_size = int(args.get("batch_size", 0) or 0)
    seq_len = int(args.get("seq_len", 0) or 0)
    if steps <= 0 or batch_size <= 0 or seq_len <= 0:
        return 0.0
    return float(steps * batch_size * seq_len)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robust multi-seed harness for distance-banded attention.")
    p.add_argument("--seeds", type=str, default="11,22,33,44,55")
    p.add_argument(
        "--modes",
        type=str,
        default="baseline,distance_prefix,distance_per_band",
        help="Comma-separated mode list. First mode is treated as baseline for paired deltas.",
    )
    p.add_argument("--metric", type=str, default="answer_acc")
    p.add_argument("--n-bootstrap", type=int, default=5000, dest="n_bootstrap")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--numpy-seed", type=int, default=12345, dest="numpy_seed")
    p.add_argument("--python-bin", type=str, default=sys.executable, dest="python_bin")
    p.add_argument(
        "--experiment-script",
        type=str,
        default="distance_band_experiment.py",
        dest="experiment_script",
    )
    p.add_argument("--out-dir", type=str, default="runs/robust/raw", dest="out_dir")
    p.add_argument("--summary-json", type=str, default="runs/robust/summary.json", dest="summary_json")
    p.add_argument("--summary-md", type=str, default="runs/robust/summary.md", dest="summary_md")
    p.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Extra args forwarded to distance_band_experiment.py.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    seeds = parse_int_list(args.seeds)
    modes = parse_str_list(args.modes)
    metric = args.metric

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_md = Path(args.summary_md)
    summary_md.parent.mkdir(parents=True, exist_ok=True)

    exp_script = Path(args.experiment_script)
    if not exp_script.exists():
        raise FileNotFoundError(f"Experiment script not found: {exp_script}")

    extra_args = shlex.split(args.extra_args)

    raw: Dict[str, Dict[int, Dict[str, float]]] = {m: {} for m in modes}
    run_stats: Dict[str, Dict[int, Dict[str, float]]] = {m: {} for m in modes}
    for seed in seeds:
        for mode in modes:
            run_json, runtime_seconds = run_once(
                python_bin=args.python_bin,
                experiment_script=exp_script,
                mode=mode,
                seed=seed,
                out_dir=out_dir,
                extra_args=extra_args,
            )
            payload = json.loads(run_json.read_text())
            raw[mode][seed] = load_run_json(run_json, mode=mode)
            train_tokens_seen = infer_train_tokens_seen(payload)
            train_tokens_per_sec = (
                train_tokens_seen / runtime_seconds if runtime_seconds > 0 and train_tokens_seen > 0 else 0.0
            )
            run_stats[mode][seed] = {
                "runtime_seconds": float(runtime_seconds),
                "train_tokens_seen": float(train_tokens_seen),
                "train_tokens_per_sec": float(train_tokens_per_sec),
            }

    rng = np.random.default_rng(args.numpy_seed)

    mode_summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_metrics = sorted({k for m in modes for s in seeds for k in raw[m][s].keys()})
    for mode in modes:
        mode_summaries[mode] = {}
        for m in all_metrics:
            vals = [float(raw[mode][s][m]) for s in seeds if m in raw[mode][s]]
            if not vals:
                continue
            stats = summarize_metric(
                vals,
                n_bootstrap=args.n_bootstrap,
                alpha=args.alpha,
                rng=rng,
            )
            mode_summaries[mode][m] = {
                "n": float(stats.n),
                "mean": stats.mean,
                "std": stats.std,
                "ci_low": stats.ci_low,
                "ci_high": stats.ci_high,
                "min": stats.min_val,
                "max": stats.max_val,
            }
        runtime_vals = [run_stats[mode][s]["runtime_seconds"] for s in seeds if s in run_stats[mode]]
        if runtime_vals:
            rstats = summarize_metric(
                runtime_vals,
                n_bootstrap=args.n_bootstrap,
                alpha=args.alpha,
                rng=rng,
            )
            mode_summaries[mode]["runtime_seconds"] = {
                "n": float(rstats.n),
                "mean": rstats.mean,
                "std": rstats.std,
                "ci_low": rstats.ci_low,
                "ci_high": rstats.ci_high,
                "min": rstats.min_val,
                "max": rstats.max_val,
            }
        tps_vals = [run_stats[mode][s]["train_tokens_per_sec"] for s in seeds if s in run_stats[mode]]
        if tps_vals:
            tstats = summarize_metric(
                tps_vals,
                n_bootstrap=args.n_bootstrap,
                alpha=args.alpha,
                rng=rng,
            )
            mode_summaries[mode]["train_tokens_per_sec"] = {
                "n": float(tstats.n),
                "mean": tstats.mean,
                "std": tstats.std,
                "ci_low": tstats.ci_low,
                "ci_high": tstats.ci_high,
                "min": tstats.min_val,
                "max": tstats.max_val,
            }
        token_vals = [run_stats[mode][s]["train_tokens_seen"] for s in seeds if s in run_stats[mode]]
        if token_vals:
            kstats = summarize_metric(
                token_vals,
                n_bootstrap=args.n_bootstrap,
                alpha=args.alpha,
                rng=rng,
            )
            mode_summaries[mode]["train_tokens_seen"] = {
                "n": float(kstats.n),
                "mean": kstats.mean,
                "std": kstats.std,
                "ci_low": kstats.ci_low,
                "ci_high": kstats.ci_high,
                "min": kstats.min_val,
                "max": kstats.max_val,
            }

    baseline_mode = modes[0]
    paired: Dict[str, Dict[str, float]] = {}
    paired_runtime: Dict[str, Dict[str, float]] = {}
    for mode in modes[1:]:
        common_seeds = [s for s in seeds if s in raw[baseline_mode] and s in raw[mode]]
        deltas = [
            float(raw[mode][s][metric]) - float(raw[baseline_mode][s][metric])
            for s in common_seeds
        ]
        if not deltas:
            continue
        delta_stats = summarize_metric(
            deltas,
            n_bootstrap=args.n_bootstrap,
            alpha=args.alpha,
            rng=rng,
        )
        sign = sign_test_two_sided(deltas)
        paired[mode] = {
            "baseline_mode": baseline_mode,
            "target_mode": mode,
            "metric": metric,
            "n": float(delta_stats.n),
            "mean_delta": delta_stats.mean,
            "std_delta": delta_stats.std,
            "ci_low": delta_stats.ci_low,
            "ci_high": delta_stats.ci_high,
            "wins": sign["wins"],
            "losses": sign["losses"],
            "ties": sign["ties"],
            "n_effective": sign["n_effective"],
            "win_rate": sign["win_rate"],
            "sign_test_p_value": sign["p_value"],
        }

        runtime_common = [s for s in seeds if s in run_stats[baseline_mode] and s in run_stats[mode]]
        if runtime_common:
            runtime_deltas = [
                run_stats[mode][s]["runtime_seconds"] - run_stats[baseline_mode][s]["runtime_seconds"]
                for s in runtime_common
            ]
            speedups = [
                run_stats[baseline_mode][s]["runtime_seconds"] / run_stats[mode][s]["runtime_seconds"]
                for s in runtime_common
                if run_stats[mode][s]["runtime_seconds"] > 0
            ]
            tps_ratios = [
                run_stats[mode][s]["train_tokens_per_sec"] / run_stats[baseline_mode][s]["train_tokens_per_sec"]
                for s in runtime_common
                if run_stats[baseline_mode][s]["train_tokens_per_sec"] > 0
            ]
            if not speedups or not tps_ratios:
                continue
            runtime_delta_stats = summarize_metric(
                runtime_deltas,
                n_bootstrap=args.n_bootstrap,
                alpha=args.alpha,
                rng=rng,
            )
            speedup_stats = summarize_metric(
                speedups,
                n_bootstrap=args.n_bootstrap,
                alpha=args.alpha,
                rng=rng,
            )
            tps_ratio_stats = summarize_metric(
                tps_ratios,
                n_bootstrap=args.n_bootstrap,
                alpha=args.alpha,
                rng=rng,
            )
            runtime_sign = sign_test_two_sided([-x for x in runtime_deltas])  # faster => negative delta
            paired_runtime[mode] = {
                "baseline_mode": baseline_mode,
                "target_mode": mode,
                "n": float(runtime_delta_stats.n),
                "mean_runtime_delta_seconds": runtime_delta_stats.mean,
                "std_runtime_delta_seconds": runtime_delta_stats.std,
                "runtime_delta_ci_low": runtime_delta_stats.ci_low,
                "runtime_delta_ci_high": runtime_delta_stats.ci_high,
                "mean_speedup_x": speedup_stats.mean,
                "speedup_ci_low": speedup_stats.ci_low,
                "speedup_ci_high": speedup_stats.ci_high,
                "mean_tps_ratio": tps_ratio_stats.mean,
                "tps_ratio_ci_low": tps_ratio_stats.ci_low,
                "tps_ratio_ci_high": tps_ratio_stats.ci_high,
                "faster_wins": runtime_sign["wins"],
                "faster_losses": runtime_sign["losses"],
                "ties": runtime_sign["ties"],
                "faster_win_rate": runtime_sign["win_rate"],
                "sign_test_p_value": runtime_sign["p_value"],
            }

    payload = {
        "seeds": seeds,
        "modes": modes,
        "metric": metric,
        "alpha": args.alpha,
        "n_bootstrap": args.n_bootstrap,
        "extra_args": extra_args,
        "mode_summaries": mode_summaries,
        "paired_deltas": paired,
        "paired_runtime": paired_runtime,
        "run_stats": run_stats,
        "raw": raw,
    }
    summary_json.write_text(json.dumps(payload, indent=2))

    lines: List[str] = []
    lines.append("# Robust Experiment Summary")
    lines.append("")
    lines.append(f"- Seeds: `{','.join(str(s) for s in seeds)}`")
    lines.append(f"- Modes: `{','.join(modes)}`")
    lines.append(f"- Metric: `{metric}`")
    lines.append(f"- Baseline mode for paired deltas: `{baseline_mode}`")
    lines.append("")
    lines.append("## Mode Means")
    lines.append("")
    lines.append("| mode | mean | std | 95% CI |")
    lines.append("|---|---:|---:|---:|")
    for mode in modes:
        ms = mode_summaries.get(mode, {}).get(metric)
        if ms is None:
            continue
        ci = f"[{ms['ci_low']:.6f}, {ms['ci_high']:.6f}]"
        lines.append(f"| {mode} | {ms['mean']:.6f} | {ms['std']:.6f} | {ci} |")

    lines.append("")
    lines.append("## Runtime And Throughput")
    lines.append("")
    lines.append("| mode | runtime_s mean | runtime_s std | runtime_s 95% CI | train tok/s mean | train tok/s 95% CI |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for mode in modes:
        rs = mode_summaries.get(mode, {}).get("runtime_seconds")
        ts = mode_summaries.get(mode, {}).get("train_tokens_per_sec")
        if rs is None or ts is None:
            continue
        rci = f"[{rs['ci_low']:.3f}, {rs['ci_high']:.3f}]"
        tci = f"[{ts['ci_low']:.1f}, {ts['ci_high']:.1f}]"
        lines.append(
            f"| {mode} | {rs['mean']:.3f} | {rs['std']:.3f} | {rci} | {ts['mean']:.1f} | {tci} |"
        )

    lines.append("")
    lines.append("## Paired Deltas Vs Baseline")
    lines.append("")
    lines.append("| target | mean_delta | std_delta | 95% CI | win_rate | sign_test_p |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for mode in modes[1:]:
        ps = paired.get(mode)
        if ps is None:
            continue
        ci = f"[{ps['ci_low']:.6f}, {ps['ci_high']:.6f}]"
        lines.append(
            f"| {mode} | {ps['mean_delta']:.6f} | {ps['std_delta']:.6f} | "
            f"{ci} | {ps['win_rate']:.3f} | {ps['sign_test_p_value']:.4f} |"
        )

    lines.append("")
    lines.append("## Runtime Vs Baseline")
    lines.append("")
    lines.append("| target | mean_runtime_delta_s | runtime_delta 95% CI | mean_speedup_x | speedup 95% CI | faster_win_rate | sign_test_p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for mode in modes[1:]:
        pr = paired_runtime.get(mode)
        if pr is None:
            continue
        dci = f"[{pr['runtime_delta_ci_low']:.3f}, {pr['runtime_delta_ci_high']:.3f}]"
        sci = f"[{pr['speedup_ci_low']:.3f}, {pr['speedup_ci_high']:.3f}]"
        lines.append(
            f"| {mode} | {pr['mean_runtime_delta_seconds']:+.3f} | {dci} | "
            f"{pr['mean_speedup_x']:.3f} | {sci} | {pr['faster_win_rate']:.3f} | {pr['sign_test_p_value']:.4f} |"
        )

    summary_md.write_text("\n".join(lines) + "\n")

    print("\n=== Robust Summary (metric means) ===")
    for mode in modes:
        ms = mode_summaries.get(mode, {}).get(metric)
        if ms is None:
            continue
        print(
            f"{mode:18s} mean={ms['mean']:.6f} std={ms['std']:.6f} "
            f"ci=[{ms['ci_low']:.6f}, {ms['ci_high']:.6f}]"
        )

    if paired:
        print(f"\n=== Paired Deltas Vs {baseline_mode} ({metric}) ===")
        for mode in modes[1:]:
            ps = paired.get(mode)
            if ps is None:
                continue
            print(
                f"{mode:18s} delta={ps['mean_delta']:+.6f} "
                f"ci=[{ps['ci_low']:+.6f}, {ps['ci_high']:+.6f}] "
                f"win_rate={ps['win_rate']:.3f} p={ps['sign_test_p_value']:.4f}"
            )

    print("\n=== Runtime Summary ===")
    for mode in modes:
        rs = mode_summaries.get(mode, {}).get("runtime_seconds")
        ts = mode_summaries.get(mode, {}).get("train_tokens_per_sec")
        if rs is None or ts is None:
            continue
        print(
            f"{mode:18s} runtime_s={rs['mean']:.3f} "
            f"ci=[{rs['ci_low']:.3f}, {rs['ci_high']:.3f}] "
            f"tok/s={ts['mean']:.1f}"
        )

    if paired_runtime:
        print(f"\n=== Runtime Vs {baseline_mode} ===")
        for mode in modes[1:]:
            pr = paired_runtime.get(mode)
            if pr is None:
                continue
            print(
                f"{mode:18s} runtime_delta_s={pr['mean_runtime_delta_seconds']:+.3f} "
                f"speedup_x={pr['mean_speedup_x']:.3f} "
                f"faster_win_rate={pr['faster_win_rate']:.3f} "
                f"p={pr['sign_test_p_value']:.4f}"
            )

    print(f"\nSaved summary JSON: {summary_json}")
    print(f"Saved summary MD:   {summary_md}")


if __name__ == "__main__":
    main()

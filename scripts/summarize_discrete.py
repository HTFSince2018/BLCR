from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


CLASSIC_MINMAX = {
    "acrobot_v1": (-500.0, -100.0),
    "Acrobot-v1": (-500.0, -100.0),
    "mountaincar_v0": (-1000.0, -100.0),
    "MountainCar-v0": (-1000.0, -100.0),
    "cartpole_v0": (0.0, 200.0),
    "CartPole-v0": (0.0, 200.0),
    "cartpole_v1": (0.0, 500.0),
    "CartPole-v1": (0.0, 500.0),
    "lunarlander_v3": (-500.0, 200.0),
    "LunarLander-v3": (-500.0, 200.0),
}


def iqm(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    ordered = np.sort(array)
    lo = int(math.ceil(0.25 * ordered.size))
    hi = int(math.floor(0.75 * ordered.size))
    if hi <= lo:
        return float(np.median(ordered))
    return float(np.mean(ordered[lo:hi]))


def task_method_seed_from_path(result_path: Path, logdir: Path) -> tuple[str, str, str]:
    rel = result_path.relative_to(logdir)
    parts = rel.parts
    if len(parts) >= 5:
        # Typical queue layout: method/task/method/seedN/result.pt.
        method, task, _algo, seed = parts[0], parts[1], parts[2], parts[3]
        return task, method, seed.replace("seed", "")
    if len(parts) >= 4:
        # Direct trainer layout: task/method/seedN/result.pt.
        task, method, seed = parts[0], parts[1], parts[2]
        return task, method, seed.replace("seed", "")
    raise ValueError(f"Cannot infer task/method/seed from {result_path}")


def load_returns(result_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(result_path, map_location="cpu", weights_only=False)
    returns = np.asarray(payload.get("returns", []), dtype=np.float64)
    frames = np.asarray(payload.get("frame_stamps", []), dtype=np.int64)
    return returns, frames


def tail_mean(returns: np.ndarray, n: int) -> float:
    if returns.size == 0:
        return float("nan")
    return float(np.mean(returns[-min(n, returns.size) :]))


def moving_mean(returns: np.ndarray, n: int) -> np.ndarray:
    if returns.size == 0:
        return np.asarray([], dtype=np.float64)
    if returns.size < n:
        return np.asarray([float(np.mean(returns))], dtype=np.float64)
    kernel = np.ones(n, dtype=np.float64) / n
    return np.convolve(returns, kernel, mode="valid")


def normalize_score(task: str, value: float, minmax: dict[str, tuple[float, float]]) -> float:
    if not np.isfinite(value):
        return float("nan")
    if task not in minmax:
        return value
    low, high = minmax[task]
    if high <= low:
        return value
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, type=Path)
    parser.add_argument("--window", default=100, type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--classic-minmax",
        action="store_true",
        help="Use the min-max intervals reported in the Neurocomputing classic-control section.",
    )
    args = parser.parse_args()

    logdir = args.logdir
    output_dir = args.output_dir or (logdir / "_launcher")
    output_dir.mkdir(parents=True, exist_ok=True)

    minmax = CLASSIC_MINMAX if args.classic_minmax else {}
    run_rows: list[dict[str, object]] = []
    for result_path in sorted(logdir.rglob("result.pt")):
        task, method, seed = task_method_seed_from_path(result_path, logdir)
        returns, frames = load_returns(result_path)
        smoothed = moving_mean(returns, args.window)
        score = tail_mean(returns, args.window)
        normalized = normalize_score(task, score, minmax)
        run_rows.append(
            {
                "task": task,
                "method": method,
                "seed": seed,
                "episodes": int(returns.size),
                "frames": int(frames[-1]) if frames.size else 0,
                f"last{args.window}_mean": score,
                f"best{args.window}_mean": float(np.max(smoothed)) if smoothed.size else float("nan"),
                "last_return": float(returns[-1]) if returns.size else float("nan"),
                "normalized_score": normalized,
                "result_path": str(result_path),
            }
        )

    if not run_rows:
        print(f"No result.pt files found under {logdir}")
        return

    per_run_path = output_dir / "paper_metrics_per_run.csv"
    with per_run_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(run_rows)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["task"]), str(row["method"]))].append(row)

    agg_rows: list[dict[str, object]] = []
    score_key = f"last{args.window}_mean"
    best_key = f"best{args.window}_mean"
    for (task, method), rows in sorted(grouped.items()):
        scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
        best_scores = np.asarray([float(row[best_key]) for row in rows], dtype=np.float64)
        normalized = np.asarray([float(row["normalized_score"]) for row in rows], dtype=np.float64)
        agg_rows.append(
            {
                "task": task,
                "method": method,
                "seeds": len(rows),
                f"{score_key}_mean": float(np.nanmean(scores)),
                f"{score_key}_std": float(np.nanstd(scores)),
                f"{score_key}_median": float(np.nanmedian(scores)),
                f"{score_key}_iqm": iqm(scores),
                f"{best_key}_mean": float(np.nanmean(best_scores)),
                "normalized_mean": float(np.nanmean(normalized)),
                "normalized_median": float(np.nanmedian(normalized)),
                "normalized_iqm": iqm(normalized),
                "optimality_gap": float(1.0 - np.nanmean(normalized)) if minmax else float("nan"),
            }
        )

    aggregate_path = output_dir / "paper_metrics_aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)

    profile_rows: list[dict[str, object]] = []
    if minmax:
        thresholds = np.linspace(0.0, 1.0, 21)
        by_method: dict[str, list[float]] = defaultdict(list)
        for row in run_rows:
            by_method[str(row["method"])].append(float(row["normalized_score"]))
        for method, values in sorted(by_method.items()):
            values_arr = np.asarray(values, dtype=np.float64)
            values_arr = values_arr[np.isfinite(values_arr)]
            for threshold in thresholds:
                profile_rows.append(
                    {
                        "method": method,
                        "threshold": float(threshold),
                        "fraction_above_threshold": float(np.mean(values_arr > threshold))
                        if values_arr.size
                        else float("nan"),
                    }
                )
        profile_path = output_dir / "paper_metrics_performance_profile.csv"
        with profile_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(profile_rows[0].keys()))
            writer.writeheader()
            writer.writerows(profile_rows)

    for row in agg_rows:
        print(
            f"{row['task']:18s} {row['method']:16s} seeds={row['seeds']} "
            f"mean={row[f'{score_key}_mean']:.3f} "
            f"std={row[f'{score_key}_std']:.3f} "
            f"iqm={row[f'{score_key}_iqm']:.3f}"
        )
    print(f"Saved per-run metrics to {per_run_path}")
    print(f"Saved aggregate metrics to {aggregate_path}")
    if profile_rows:
        print(f"Saved performance profile data to {output_dir / 'paper_metrics_performance_profile.csv'}")


if __name__ == "__main__":
    main()

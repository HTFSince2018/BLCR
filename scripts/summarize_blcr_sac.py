from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_evals(metrics_path: Path) -> list[dict]:
    rows = []
    if not metrics_path.exists():
        return rows
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "eval":
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs_blcr_sac")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    logdir = Path(args.logdir)
    out_dir = Path(args.out_dir) if args.out_dir else logdir / "_launcher"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run = []
    for metrics_path in logdir.glob("*/*/seed*/metrics.jsonl"):
        evals = read_evals(metrics_path)
        if not evals:
            continue
        method = metrics_path.parts[-4]
        env = metrics_path.parts[-3]
        seed = metrics_path.parts[-2].replace("seed", "")
        last = evals[-1]
        best = max(evals, key=lambda x: x.get("eval_return_mean", float("-inf")))
        per_run.append(
            {
                "env": env,
                "method": method,
                "seed": seed,
                "last_step": last.get("step", ""),
                "last_return_mean": last.get("eval_return_mean", np.nan),
                "last_return_std": last.get("eval_return_std", np.nan),
                "best_step": best.get("step", ""),
                "best_return_mean": best.get("eval_return_mean", np.nan),
                "fps": last.get("fps", np.nan),
                "metrics_path": str(metrics_path),
            }
        )

    per_run_path = out_dir / "blcr_sac_per_run.csv"
    with per_run_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run[0].keys()) if per_run else ["env", "method", "seed"])
        writer.writeheader()
        writer.writerows(per_run)

    grouped = defaultdict(list)
    for row in per_run:
        grouped[(row["env"], row["method"])].append(row)

    aggregate = []
    for (env, method), rows in sorted(grouped.items()):
        last_vals = np.asarray([float(r["last_return_mean"]) for r in rows], dtype=np.float64)
        best_vals = np.asarray([float(r["best_return_mean"]) for r in rows], dtype=np.float64)
        aggregate.append(
            {
                "env": env,
                "method": method,
                "n": len(rows),
                "last_return_mean": float(np.mean(last_vals)),
                "last_return_std": float(np.std(last_vals)),
                "last_return_median": float(np.median(last_vals)),
                "best_return_mean": float(np.mean(best_vals)),
                "best_return_std": float(np.std(best_vals)),
                "mean_fps": float(np.nanmean([float(r["fps"]) for r in rows])),
            }
        )

    aggregate_path = out_dir / "blcr_sac_aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()) if aggregate else ["env", "method", "n"])
        writer.writeheader()
        writer.writerows(aggregate)

    print(f"wrote {per_run_path}")
    print(f"wrote {aggregate_path}")


if __name__ == "__main__":
    main()

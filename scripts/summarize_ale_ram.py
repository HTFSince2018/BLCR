from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev

import torch


def window_mean(values: list[float], n: int) -> float:
    if not values:
        return math.nan
    return float(mean(values[-min(n, len(values)) :]))


def best_window_mean(values: list[float], n: int) -> float:
    if not values:
        return math.nan
    width = min(n, len(values))
    return float(max(mean(values[i : i + width]) for i in range(0, len(values) - width + 1)))


def load_run(path: Path) -> dict[str, object]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    returns = [float(x) for x in data.get("returns", [])]
    frame_stamps = data.get("frame_stamps", [])
    seed_dir = path.parent
    seed = int(seed_dir.name.replace("seed", ""))
    method = seed_dir.parent.name
    task = seed_dir.parent.parent.name
    return {
        "method": method,
        "task": task,
        "seed": seed,
        "episodes": len(returns),
        "frames": int(frame_stamps[-1]) if frame_stamps else 0,
        "last10": window_mean(returns, 10),
        "last50": window_mean(returns, 50),
        "last100": window_mean(returns, 100),
        "best10": best_window_mean(returns, 10),
        "best50": best_window_mean(returns, 50),
        "best100": best_window_mean(returns, 100),
        "last_return": returns[-1] if returns else math.nan,
        "path": str(path),
    }


def fmt(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.6g}"


def summarize_metric(rows: list[dict[str, object]], metric: str) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        task = str(row["task"])
        method = str(row["method"])
        grouped.setdefault(task, {}).setdefault(method, []).append(float(row[metric]))

    out = []
    for task, by_method in sorted(grouped.items()):
        if "relo" not in by_method or "blcr" not in by_method:
            continue
        relo_values = by_method["relo"]
        ours_values = by_method["blcr"]
        relo_mean = float(mean(relo_values))
        ours_mean = float(mean(ours_values))
        diff = ours_mean - relo_mean
        ratio = ours_mean / relo_mean if abs(relo_mean) > 1e-12 else math.nan
        paired = []
        for seed in sorted({int(r["seed"]) for r in rows if str(r["task"]) == task}):
            r = [float(x[metric]) for x in rows if str(x["task"]) == task and str(x["method"]) == "relo" and int(x["seed"]) == seed]
            o = [
                float(x[metric])
                for x in rows
                if str(x["task"]) == task
                and str(x["method"]) == "blcr"
                and int(x["seed"]) == seed
            ]
            if r and o:
                paired.append(o[0] - r[0])
        out.append(
            {
                "task": task,
                "metric": metric,
                "relo_mean": relo_mean,
                "ours_mean": ours_mean,
                "diff": diff,
                "ratio": ratio,
                "relo_std": float(pstdev(relo_values)) if len(relo_values) > 1 else 0.0,
                "ours_std": float(pstdev(ours_values)) if len(ours_values) > 1 else 0.0,
                "seed_wins": sum(1 for x in paired if x > 0),
                "paired_count": len(paired),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs_ale_ram_30games")
    parser.add_argument("--metric", default="last100", choices=["last10", "last50", "last100", "best10", "best50", "best100", "last_return"])
    args = parser.parse_args()

    root = Path(args.logdir)
    result_paths = sorted(root.glob("*/*/*/seed*/result.pt"))
    rows = [load_run(path) for path in result_paths]
    out_dir = root / "_launcher"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / "final_summary_full.csv"
    with full_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    task_rows = summarize_metric(rows, args.metric)
    task_path = out_dir / f"final_summary_by_task_{args.metric}.csv"
    with task_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()))
        writer.writeheader()
        writer.writerows(task_rows)

    wins = [r for r in task_rows if float(r["diff"]) > 0]
    losses = [r for r in task_rows if float(r["diff"]) < 0]
    print(f"runs={len(rows)} tasks={len(task_rows)} metric={args.metric}")
    print(f"ours_wins={len(wins)} relo_wins={len(losses)} ties={len(task_rows)-len(wins)-len(losses)}")
    print(f"mean_diff={fmt(mean(float(r['diff']) for r in task_rows))}")
    ratios = [float(r["ratio"]) for r in task_rows if math.isfinite(float(r["ratio"]))]
    print(f"mean_ratio={fmt(mean(ratios))}")
    print(f"full_csv={full_path}")
    print(f"task_csv={task_path}")


if __name__ == "__main__":
    main()

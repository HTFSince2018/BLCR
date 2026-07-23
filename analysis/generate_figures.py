from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"


COLORS = {
    "DQN": "#4C78A8",
    "PER": "#F58518",
    "ReLo": "#54A24B",
    "ReLo-F": "#B279A2",
    "BLCR": "#E45756",
    "SAC": "#4C78A8",
    "PER-SAC": "#F58518",
    "ReLo-SAC": "#54A24B",
    "BLCR-SAC": "#E45756",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=220)
    plt.close(fig)


def read_environment_frame(filename: str) -> np.ndarray:
    path = FIG_DIR / "env_frames" / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run make_environment_frames.py before generating figures.")
    image = plt.imread(path)
    if image.shape[0] > 40:
        image = image[:-30, ...]  # Remove the embedded label; figure text handles labeling.
    return image


def display_name(task: str) -> str:
    name = task
    name = name.replace("ale_", "").replace("_ram", "")
    parts = name.split("_")
    overrides = {
        "ms_pacman": "Ms. Pac-Man",
        "jamesbond": "James Bond",
    }
    key = "_".join(parts)
    if key in overrides:
        return overrides[key]
    return " ".join(p.capitalize() for p in parts)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < 3:
        return y
    window = min(window, len(y))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return y
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def interpolate_runs(runs: list[tuple[np.ndarray, np.ndarray]], n_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs = [(x, y) for x, y in runs if len(x) >= 2 and np.nanmax(x) > np.nanmin(x)]
    if not runs:
        return np.array([]), np.array([]), np.array([])
    min_end = min(float(np.nanmax(x)) for x, _ in runs)
    max_start = max(float(np.nanmin(x)) for x, _ in runs)
    if min_end <= max_start:
        max_start = min(float(np.nanmin(x)) for x, _ in runs)
    grid = np.linspace(max_start, min_end, n_points)
    values = []
    for x, y in runs:
        order = np.argsort(x)
        xx = x[order]
        yy = y[order]
        keep = np.isfinite(xx) & np.isfinite(yy)
        xx = xx[keep]
        yy = yy[keep]
        if len(xx) < 2:
            continue
        values.append(np.interp(grid, xx, yy))
    if not values:
        return np.array([]), np.array([]), np.array([])
    matrix = np.vstack(values)
    return grid, np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0)


def metric_runs_discrete(root: Path, method: str, task: str, y_key: str = "avg_return") -> list[tuple[np.ndarray, np.ndarray]]:
    runs: list[tuple[np.ndarray, np.ndarray]] = []
    task_dir = root / method / task / method
    for metrics in sorted(task_dir.glob("seed*/metrics.jsonl")):
        rows = read_jsonl(metrics)
        xs = np.array([float(r["step"]) for r in rows if "step" in r and y_key in r], dtype=float)
        ys = np.array([float(r[y_key]) for r in rows if "step" in r and y_key in r], dtype=float)
        if len(xs) >= 2:
            runs.append((xs, smooth(ys, 9)))
    return runs


def metric_runs_sac(root: Path, method: str, task: str) -> list[tuple[np.ndarray, np.ndarray]]:
    runs: list[tuple[np.ndarray, np.ndarray]] = []
    task_dir = root / method / task
    for metrics in sorted(task_dir.glob("seed*/metrics.jsonl")):
        rows = [
            r
            for r in read_jsonl(metrics)
            if r.get("type") == "eval" and "step" in r and "eval_return_mean" in r
        ]
        xs = np.array([float(r["step"]) for r in rows], dtype=float)
        ys = np.array([float(r["eval_return_mean"]) for r in rows], dtype=float)
        if len(xs) >= 2:
            runs.append((xs, smooth(ys, 3)))
    return runs


def plot_method_framework() -> None:
    fig, ax = plt.subplots(figsize=(7.05, 2.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x: float, y: float, w: float, h: float, text: str, fc: str = "#F8F8F8", ec: str = "#333333") -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=0.8,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)

    def arrow(xy1: tuple[float, float], xy2: tuple[float, float], color: str = "#555555") -> None:
        ax.add_patch(
            FancyArrowPatch(
                xy1,
                xy2,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.8,
                color=color,
                shrinkA=2,
                shrinkB=2,
            )
        )

    box(0.035, 0.39, 0.12, 0.2, "Replay\nbuffer")
    box(0.20, 0.39, 0.13, 0.2, "Mini-batch\ntransitions")
    box(0.39, 0.66, 0.16, 0.17, "Online loss\n$L_i^{on}$")
    box(0.39, 0.42, 0.16, 0.17, "Reference loss\n$L_i^{ref}$")
    box(0.39, 0.18, 0.16, 0.17, "TD magnitude\n$D_i$")
    box(0.61, 0.56, 0.14, 0.18, "Reducible\n$R_i=[L_i^{on}-L_i^{ref}]_+$", "#FFF7F7", "#A63B3B")
    box(0.61, 0.23, 0.14, 0.18, "TD floor\n$f_i=\\lambda\\log(1+\\hat D_i)$", "#F5FAFF", "#2F5D8C")
    box(0.79, 0.42, 0.17, 0.2, "$p_i=\\max\\{\\hat R_i,f_i\\}$\npriority")
    box(0.765, 0.10, 0.215, 0.18, "$g_i=\\mathrm{clip}(1+\\eta\\log(1+\\hat R_i))$\nloss weight")

    arrow((0.155, 0.49), (0.20, 0.49))
    arrow((0.33, 0.50), (0.39, 0.75))
    arrow((0.33, 0.50), (0.39, 0.51))
    arrow((0.33, 0.48), (0.39, 0.27))
    arrow((0.55, 0.735), (0.61, 0.65), "#A63B3B")
    arrow((0.55, 0.505), (0.61, 0.65), "#A63B3B")
    arrow((0.55, 0.265), (0.61, 0.32), "#2F5D8C")
    arrow((0.75, 0.64), (0.79, 0.52))
    arrow((0.75, 0.32), (0.79, 0.50))
    arrow((0.875, 0.42), (0.875, 0.28))
    arrow((0.875, 0.62), (0.15, 0.60), "#777777")

    ax.text(0.875, 0.68, "sample next updates", ha="center", va="bottom", fontsize=6.8, color="#555555")
    ax.text(0.875, 0.035, "bounded update strength", ha="center", va="bottom", fontsize=6.8, color="#555555")
    ax.text(0.06, 0.17, "DQN: target-network reference\nSAC: one-step critic loss reduction", fontsize=6.8, color="#555555")
    save_figure(fig, "fig_blcr_framework")


def plot_benchmark_overview() -> None:
    panels = [
        (
            "MinAtar Breakout",
            "minatar_breakout.png",
            "10x10 visual state\n5 games, 2M frames, 5 seeds",
        ),
        (
            "ALE Asterix",
            "ale_asterix.png",
            "rendered emulator frame\nagent observes 128-byte RAM",
        ),
        (
            "MuJoCo Hopper",
            "mujoco_hopper.png",
            "offscreen simulator frame\n4 tasks, 300K steps, 3 seeds",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.35))
    for ax, (title, filename, subtitle) in zip(axes, panels):
        image = read_environment_frame(filename)
        ax.imshow(image)
        ax.set_title(title, pad=4)
        ax.set_axis_off()
        ax.text(
            0.5,
            -0.10,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.0,
            color="#444444",
            linespacing=1.15,
        )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.86, bottom=0.26, wspace=0.16)
    save_figure(fig, "fig_real_environment_frames_wide")


def plot_environment_frames_column() -> None:
    panels = [
        (
            "MinAtar Breakout",
            "minatar_breakout.png",
            "compact 10x10 visual state",
        ),
        (
            "ALE Asterix",
            "ale_asterix.png",
            "rendered emulator frame\nagent observes 128-byte RAM",
        ),
        (
            "MuJoCo Hopper",
            "mujoco_hopper.png",
            "continuous-control simulator frame",
        ),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 4.05))
    for ax, (title, filename, subtitle) in zip(axes, panels):
        ax.imshow(read_environment_frame(filename))
        ax.set_title(title, pad=2)
        ax.set_axis_off()
        ax.text(
            0.5,
            -0.08,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.0,
            color="#444444",
        )
    fig.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.04, hspace=0.48)
    save_figure(fig, "fig_real_environment_frames")


def plot_minatar_curves() -> None:
    root = REPO_ROOT / "logs_minatar_neurocomputing_2m"
    tasks = ["asterix", "breakout", "freeway", "seaquest", "space_invaders"]
    methods = [
        ("dqn", "DQN"),
        ("per", "PER"),
        ("relo", "ReLo"),
        ("relo_floor", "ReLo-F"),
        ("blcr", "BLCR"),
    ]
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.05, 1.75), sharex=True)
    for ax, task in zip(axes, tasks):
        for method, label in methods:
            x, mean, std = interpolate_runs(metric_runs_discrete(root, method, task), 180)
            if len(x) == 0:
                continue
            ax.plot(x / 1e6, mean, label=label, color=COLORS[label], lw=1.1)
            ax.fill_between(x / 1e6, mean - std, mean + std, color=COLORS[label], alpha=0.12, linewidth=0)
        ax.set_title(display_name(task))
        ax.set_xlabel("Frames (M)")
        if ax is axes[0]:
            ax.set_ylabel("Return")
        ax.set_xlim(0, 2.0)
        ax.ticklabel_format(axis="y", style="plain")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save_figure(fig, "fig_minatar_curves")


def plot_ale_curves() -> None:
    root = REPO_ROOT / "logs_ale_ram_30games"
    tasks = [
        "ale_alien_ram",
        "ale_battle_zone_ram",
        "ale_hero_ram",
        "ale_bank_heist_ram",
        "ale_freeway_ram",
        "ale_kung_fu_master_ram",
    ]
    methods = [("relo", "ReLo"), ("blcr", "BLCR")]
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 3.15), sharex=True)
    axes_flat = axes.ravel()
    for ax, task in zip(axes_flat, tasks):
        for method, label in methods:
            x, mean, std = interpolate_runs(metric_runs_discrete(root, method, task), 220)
            if len(x) == 0:
                continue
            ax.plot(x / 1e6, mean, label=label, color=COLORS[label], lw=1.0)
            ax.fill_between(x / 1e6, mean - std, mean + std, color=COLORS[label], alpha=0.13, linewidth=0)
        ax.set_title(display_name(task))
        ax.set_xlabel("Frames (M)")
        if ax in axes_flat[::3]:
            ax.set_ylabel("Return")
        ax.set_xlim(0, 2.0)
        ax.ticklabel_format(axis="y", style="plain")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, "fig_ale_curves")


def plot_mujoco_curves() -> None:
    root = REPO_ROOT / "logs_blcr_sac_locomotion_300k"
    tasks = ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
    methods = [
        ("sac", "SAC"),
        ("per_sac", "PER-SAC"),
        ("relo_sac", "ReLo-SAC"),
        ("blcr_sac", "BLCR-SAC"),
    ]
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.05, 1.85), sharex=True)
    for ax, task in zip(axes, tasks):
        for method, label in methods:
            x, mean, std = interpolate_runs(metric_runs_sac(root, method, task), 60)
            if len(x) == 0:
                continue
            ax.plot(x / 1000.0, mean, label=label, color=COLORS[label], lw=1.1)
            ax.fill_between(x / 1000.0, mean - std, mean + std, color=COLORS[label], alpha=0.12, linewidth=0)
        ax.set_title(task.replace("-v5", ""))
        ax.set_xlabel("Steps (K)")
        if ax is axes[0]:
            ax.set_ylabel("Eval return")
        ax.set_xlim(0, 300)
        ax.ticklabel_format(axis="y", style="plain")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save_figure(fig, "fig_mujoco_curves")


def plot_mujoco_curves_column() -> None:
    root = REPO_ROOT / "logs_blcr_sac_locomotion_300k"
    tasks = ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
    methods = [
        ("sac", "SAC"),
        ("per_sac", "PER-SAC"),
        ("relo_sac", "ReLo-SAC"),
        ("blcr_sac", "BLCR-SAC"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.05), sharex=True)
    axes_flat = axes.ravel()
    for ax, task in zip(axes_flat, tasks):
        for method, label in methods:
            x, mean, std = interpolate_runs(metric_runs_sac(root, method, task), 60)
            if len(x) == 0:
                continue
            ax.plot(x / 1000.0, mean, label=label, color=COLORS[label], lw=0.95)
            ax.fill_between(x / 1000.0, mean - std, mean + std, color=COLORS[label], alpha=0.11, linewidth=0)
        ax.set_title(task.replace("-v5", ""))
        ax.set_xlabel("Steps (K)")
        if ax in (axes_flat[0], axes_flat[2]):
            ax.set_ylabel("Eval return")
        ax.set_xlim(0, 300)
        ax.ticklabel_format(axis="y", style="plain")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save_figure(fig, "fig_mujoco_curves_column")


def main() -> None:
    configure_matplotlib()
    plot_method_framework()
    if (FIG_DIR / "env_frames").exists():
        plot_benchmark_overview()
        plot_environment_frames_column()
    plot_minatar_curves()
    plot_ale_curves()
    plot_mujoco_curves()
    plot_mujoco_curves_column()


if __name__ == "__main__":
    main()

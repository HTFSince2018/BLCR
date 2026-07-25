from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class _RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0


class TargetVarianceTracker:
    def __init__(self) -> None:
        self._stats: dict[bytes, _RunningMoments] = {}

    def update(self, state: np.ndarray, target_value: float) -> float:
        key = np.ascontiguousarray(state).tobytes()
        stats = self._stats.setdefault(key, _RunningMoments())
        stats.count += 1
        delta = target_value - stats.mean
        stats.mean += delta / stats.count
        delta2 = target_value - stats.mean
        stats.m2 += delta * delta2
        if stats.count < 2:
            return 0.0
        return stats.m2 / (stats.count - 1)


class CountNoveltyTracker:
    def __init__(self) -> None:
        self._counts: dict[bytes, int] = {}

    def observe(self, state: np.ndarray) -> float:
        key = np.ascontiguousarray(state).tobytes()
        count = self._counts.get(key, 0)
        self._counts[key] = count + 1
        return 1.0 / math.sqrt(count + 1.0)


def compute_scl_factor(
    target_variance: torch.Tensor,
    novelty: torch.Tensor,
    scl_eps: float,
    min_gate: float,
) -> torch.Tensor:
    factor = torch.exp(-(target_variance / (novelty + scl_eps)))
    if min_gate > 0.0:
        factor = torch.where(factor >= min_gate, factor, torch.zeros_like(factor))
    return factor


def compute_phi(
    td_error: torch.Tensor,
    target_variance: torch.Tensor,
    novelty: torch.Tensor,
    scl_eps: float,
    min_gate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    factor = compute_scl_factor(target_variance, novelty, scl_eps, min_gate)
    phi = td_error.abs() * factor
    return phi, factor

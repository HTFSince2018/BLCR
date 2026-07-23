from __future__ import annotations

import math

import numpy as np


def estimate_density(
    embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int,
    sigma: float,
    eps: float = 1e-6,
) -> float:
    if reference_embeddings.size == 0:
        return 1.0

    distances = np.linalg.norm(reference_embeddings - embedding[None, :], axis=1)
    nn_count = min(k, distances.shape[0])
    nearest = np.partition(distances, nn_count - 1)[:nn_count]
    weights = np.exp(-(nearest ** 2) / (2.0 * sigma * sigma + eps))
    return float(weights.mean() + eps)


def compute_survival_score(
    phi: float,
    density: float,
    age: int,
    gamma: float,
    decay: float,
    eps: float = 1e-6,
) -> float:
    learnability = (max(phi, 0.0) + eps) ** gamma
    return float((learnability / (density + eps)) * math.exp(-decay * max(age, 0)))

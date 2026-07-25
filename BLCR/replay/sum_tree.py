from __future__ import annotations

import numpy as np


class SumTree:
    def __init__(self, capacity: int):
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2

        self.capacity = capacity
        self.tree_capacity = tree_capacity
        self.tree = np.zeros(2 * tree_capacity, dtype=np.float32)
        self.max_priority = 1.0

    def total(self) -> float:
        return float(self.tree[1])

    def get(self, data_index: int) -> float:
        return float(self.tree[data_index + self.tree_capacity])

    def update(self, data_index: int, priority: float) -> None:
        if not np.isfinite(priority) or priority <= 0.0:
            priority = 1e-6
        index = data_index + self.tree_capacity
        delta = priority - self.tree[index]
        while index >= 1:
            self.tree[index] += delta
            index //= 2
        self.max_priority = max(self.max_priority, float(priority))

    def sample(self, mass: float) -> int:
        index = 1
        while index < self.tree_capacity:
            left = index * 2
            if mass <= self.tree[left]:
                index = left
            else:
                mass -= self.tree[left]
                index = left + 1
        return index - self.tree_capacity

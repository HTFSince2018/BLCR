from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from BLCR.replay.sum_tree import SumTree


@dataclass
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    nonterminals: torch.Tensor
    is_weights: torch.Tensor
    behavior_probs: torch.Tensor
    novelty: torch.Tensor
    target_variance: torch.Tensor
    learnability: torch.Tensor
    previous_loss: torch.Tensor
    data_indices: np.ndarray


class DensityAwarePrioritizedReplay:
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        num_actions: int,
        capacity: int,
        priority_alpha: float,
        priority_beta: float,
        priority_eps: float,
        embedding_dim: int,
        gamma: float,
        decay: float,
        density_k: int,
        density_sigma: float,
        density_sample_size: int,
        prune_candidates: int,
        device: torch.device,
        use_das: bool,
        das_interval: int = 1,
        priority_max: float = 1e6,
    ) -> None:
        self.capacity = capacity
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        self.priority_eps = priority_eps
        self.priority_max = max(float(priority_max), priority_eps)
        self.scaled_priority_max = float(
            np.power(self.priority_max + self.priority_eps, self.priority_alpha)
        )
        self.scaled_priority_max = max(self.scaled_priority_max, 1e-6)
        self.device = device
        self.num_actions = num_actions
        self.use_das = use_das
        self.das_interval = das_interval

        self.das_gamma = gamma
        self.das_decay = decay
        self.density_k = density_k
        self.density_sigma = density_sigma
        self.density_sample_size = density_sample_size
        self.prune_candidates = prune_candidates

        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.states = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.nonterminals = np.zeros(capacity, dtype=np.float32)
        self.behavior_probs = np.zeros((capacity, num_actions), dtype=np.float32)
        self.novelty = np.zeros(capacity, dtype=np.float32)
        self.target_variance = np.zeros(capacity, dtype=np.float32)
        self.learnability = np.zeros(capacity, dtype=np.float32)
        self.last_loss = np.zeros(capacity, dtype=np.float32)
        self.embeddings = np.zeros((capacity, embedding_dim), dtype=np.float32)
        self.timestamps = np.zeros(capacity, dtype=np.int64)
        self.valid = np.zeros(capacity, dtype=bool)

        self.size = 0
        self.next_index = 0
        self.step = 0
        self.tree = SumTree(capacity)

    @staticmethod
    def _sanitize_priority(priority: float, fallback: float = 1.0) -> float:
        if not np.isfinite(priority) or priority <= 0.0:
            return fallback
        return float(priority)

    def _uniform_sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data_indices = np.random.randint(0, self.size, size=batch_size, dtype=np.int64)
        probs = np.full(batch_size, 1.0 / self.size, dtype=np.float32)
        weights = np.ones(batch_size, dtype=np.float32)
        return data_indices, probs, weights

    def __len__(self) -> int:
        return self.size

    def anneal_beta(self, value: float) -> None:
        self.priority_beta = min(1.0, value)

    def _choose_index(self) -> int:
        if self.size < self.capacity:
            index = self.next_index
            self.next_index += 1
            return index

        if not self.use_das:
            index = self.next_index % self.capacity
            self.next_index += 1
            return index

        # Run the expensive DAS pruning only every das_interval steps;
        # the rest of the time use simple circular overwrite.
        if self.step % self.das_interval != 0:
            index = self.next_index % self.capacity
            self.next_index += 1
            return index

        # Buffer is full; all entries are valid. Vectorize DAS computation
        # to avoid O(prune_candidates * capacity) overhead.
        candidate_count = min(self.prune_candidates, self.size)
        candidates = np.random.choice(self.size, size=candidate_count, replace=False)

        pool_size = min(self.density_sample_size, self.size)
        reference_pool = np.random.choice(self.size, size=pool_size, replace=False)

        candidate_emb = self.embeddings[candidates]  # (C, D)
        ref_emb = self.embeddings[reference_pool]     # (R, D)

        # Pairwise squared distances via ||a-b||^2 = sum(a^2) + sum(b^2) - 2ab^T.
        a2 = np.sum(candidate_emb ** 2, axis=1, keepdims=True)
        b2 = np.sum(ref_emb ** 2, axis=1)
        cross = candidate_emb @ ref_emb.T
        sq_distances = np.maximum(0, a2 + b2[None, :] - 2 * cross)
        distances = np.sqrt(sq_distances)

        k = min(self.density_k, distances.shape[1])
        nearest = np.partition(distances, k - 1, axis=1)[:, :k]
        weights = np.exp(-(nearest ** 2) / (2.0 * self.density_sigma * self.density_sigma + 1e-6))
        densities = weights.mean(axis=1) + 1e-6

        ages = np.int64(self.step - self.timestamps[candidates])
        phis = np.maximum(self.learnability[candidates], 0.0)
        learnability = (phis + 1e-6) ** self.das_gamma
        scores = learnability / (densities + 1e-6) * np.exp(-self.das_decay * np.maximum(ages, 0))

        return int(candidates[np.argmin(scores)])

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        nonterminal: float,
        behavior_probs: np.ndarray,
        novelty: float,
        target_variance: float,
        learnability: float,
        embedding: np.ndarray,
    ) -> None:
        index = self._choose_index()
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.nonterminals[index] = nonterminal
        self.behavior_probs[index] = behavior_probs
        self.novelty[index] = novelty
        self.target_variance[index] = target_variance
        self.learnability[index] = learnability
        self.last_loss[index] = 0.0
        self.embeddings[index] = embedding
        self.timestamps[index] = self.step
        self.valid[index] = True

        if self.size < self.capacity:
            self.size += 1

        initial_priority = self._sanitize_priority(self.tree.max_priority, fallback=1.0)
        initial_priority = min(initial_priority, self.scaled_priority_max)
        self.tree.update(index, initial_priority)
        self.step += 1

    def _states_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.tensor(array, dtype=torch.float32, device=self.device)
        if tensor.ndim == 4:
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor

    def sample(self, batch_size: int) -> ReplayBatch:
        total = self.tree.total()
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        if not np.isfinite(total) or total <= 0.0:
            data_indices, probs, weights = self._uniform_sample(batch_size)
        else:
            masses = np.random.uniform(0.0, total, size=batch_size)
            data_indices = np.asarray(
                [self.tree.sample(mass) for mass in masses], dtype=np.int64
            )
            data_indices = np.clip(data_indices, 0, self.size - 1)
            priorities = np.asarray(
                [self.tree.get(index) for index in data_indices], dtype=np.float32
            )

            finite_positive = np.isfinite(priorities) & (priorities > 0.0)
            if not finite_positive.any():
                data_indices, probs, weights = self._uniform_sample(batch_size)
            else:
                safe_priorities = priorities.copy()
                safe_priorities[~finite_positive] = np.min(safe_priorities[finite_positive])
                probs = safe_priorities / max(float(total), 1e-12)
                probs = np.clip(probs, 1e-12, None)
                weights = np.power(self.size * probs, -self.priority_beta)
                weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
                max_weight = max(float(weights.max()), 1e-12)
                weights = weights / max_weight

        states = self._states_to_tensor(self.states[data_indices])
        next_states = self._states_to_tensor(self.next_states[data_indices])
        actions = torch.tensor(
            self.actions[data_indices], dtype=torch.int64, device=self.device
        ).unsqueeze(-1)
        rewards = torch.tensor(
            self.rewards[data_indices], dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        nonterminals = torch.tensor(
            self.nonterminals[data_indices], dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        is_weights = torch.tensor(
            weights, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        behavior_probs = torch.tensor(
            self.behavior_probs[data_indices], dtype=torch.float32, device=self.device
        )
        novelty = torch.tensor(
            self.novelty[data_indices], dtype=torch.float32, device=self.device
        )
        target_variance = torch.tensor(
            self.target_variance[data_indices], dtype=torch.float32, device=self.device
        )
        learnability = torch.tensor(
            self.learnability[data_indices], dtype=torch.float32, device=self.device
        )
        previous_loss = torch.tensor(
            self.last_loss[data_indices], dtype=torch.float32, device=self.device
        )

        return ReplayBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            nonterminals=nonterminals,
            is_weights=is_weights,
            behavior_probs=behavior_probs,
            novelty=novelty,
            target_variance=target_variance,
            learnability=learnability,
            previous_loss=previous_loss,
            data_indices=data_indices,
        )

    def update_priorities(
        self,
        data_indices: np.ndarray,
        priorities: np.ndarray,
        learnability: np.ndarray | None = None,
        loss_values: np.ndarray | None = None,
    ) -> None:
        safe_priorities = np.asarray(priorities, dtype=np.float32)
        safe_priorities = np.nan_to_num(
            safe_priorities, nan=self.priority_eps, posinf=1.0, neginf=self.priority_eps
        )
        safe_priorities = np.clip(safe_priorities, self.priority_eps, self.priority_max)
        scaled = np.power(safe_priorities + self.priority_eps, self.priority_alpha)
        scaled = np.nan_to_num(scaled, nan=1.0, posinf=1.0, neginf=1.0)
        scaled = np.clip(scaled, 1e-6, self.scaled_priority_max)
        for index, priority in zip(data_indices, scaled):
            self.tree.update(int(index), float(priority))
        if learnability is not None:
            safe_learnability = np.asarray(learnability, dtype=np.float32)
            safe_learnability = np.nan_to_num(
                safe_learnability, nan=0.0, posinf=1e6, neginf=0.0
            )
            self.learnability[data_indices] = safe_learnability
        if loss_values is not None:
            safe_losses = np.asarray(loss_values, dtype=np.float32)
            safe_losses = np.nan_to_num(safe_losses, nan=0.0, posinf=1e6, neginf=0.0)
            self.last_loss[data_indices] = np.clip(safe_losses, 0.0, 1e6)

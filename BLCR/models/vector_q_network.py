from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f


class VectorQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.q_head = nn.Linear(current_dim, num_actions)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        q_values = self.q_head(features)
        if return_features:
            return q_values, features
        return q_values

from __future__ import annotations

import torch
import torch.nn.functional as f


def compute_avgd_penalty(
    q_values: torch.Tensor,
    behavior_probs: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    behavior_probs = behavior_probs.clamp_min(eps)
    behavior_probs = behavior_probs / behavior_probs.sum(dim=-1, keepdim=True)

    current_probs = f.softmax(q_values / temperature, dim=-1).clamp_min(eps)
    with torch.no_grad():
        q_behavior = (behavior_probs * q_values).sum(dim=-1)
        q_current = (current_probs * q_values).sum(dim=-1)
        value_gate = torch.relu(q_behavior - q_current)
    kl = torch.sum(current_probs * (current_probs.log() - behavior_probs.log()), dim=-1)
    penalty = value_gate * kl

    stats = {
        "avgd_penalty": float(penalty.mean().detach().cpu()),
        "avgd_kl": float(kl.mean().detach().cpu()),
        "avgd_gate": float(value_gate.mean().detach().cpu()),
    }
    return penalty, stats

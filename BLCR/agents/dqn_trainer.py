from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as f
import torch.optim as optim
from tqdm import tqdm

from BLCR.config import load_config, parse_override, set_by_dotted_key
from BLCR.models.q_network import MinAtarQNetwork
from BLCR.modules.avgd import compute_avgd_penalty
from BLCR.modules.scl import CountNoveltyTracker, TargetVarianceTracker, compute_phi
from BLCR.replay.buffer import DensityAwarePrioritizedReplay
from BLCR.utils.logging import Logger
from BLCR.utils.seed import set_seed


def create_minatar_env(env_cls, game: str, seed: int):
    try:
        return env_cls(game, random_seed=seed)
    except TypeError:
        pass

    try:
        return env_cls(game, seed=seed)
    except TypeError:
        pass

    env = env_cls(game)
    return env


def choose_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def state_to_tensor(state: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(state, device=device).permute(2, 0, 1).unsqueeze(0).float()


def epsilon_by_frame(
    frame: int,
    replay_start_size: int,
    eps_start: float,
    eps_end: float,
    eps_decay_frames: int,
) -> float:
    if frame < replay_start_size:
        return 1.0
    elapsed = max(frame - replay_start_size, 0)
    if elapsed >= eps_decay_frames:
        return eps_end
    slope = (eps_end - eps_start) / eps_decay_frames
    return eps_start + slope * elapsed


class EmaScale:
    def __init__(self, decay: float, eps: float = 1e-6) -> None:
        self.decay = decay
        self.eps = eps
        self.scale = 1.0
        self.initialized = False

    def normalize(self, values: torch.Tensor, update: bool = True) -> torch.Tensor:
        detached = values.detach()
        if update:
            finite = detached[torch.isfinite(detached)]
            finite = finite[finite > 0.0]
            if finite.numel() > 0:
                batch_scale = max(float(finite.mean().cpu()), self.eps)
                if self.initialized:
                    self.scale = self.decay * self.scale + (1.0 - self.decay) * batch_scale
                else:
                    self.scale = batch_scale
                    self.initialized = True
        return values / max(self.scale, self.eps)


class DQNTrainer:
    def __init__(self, config: dict):
        self.config = config
        exp_cfg = config["experiment"]
        train_cfg = config["training"]
        prio_cfg = config["prioritization"]
        fluid_cfg = config["fluid"]

        self.algo = exp_cfg["algo"].lower()
        self.game = exp_cfg["game"]
        self.seed = int(exp_cfg["seed"])
        self.device = choose_device(config["system"]["device"])
        self.gamma = float(train_cfg["gamma"])
        self.batch_size = int(train_cfg["batch_size"])
        self.replay_start_size = int(train_cfg["replay_start_size"])
        self.target_update_freq = int(train_cfg["target_update_freq"])
        self.target_tau = float(train_cfg["target_tau"])
        self.train_freq = int(train_cfg["train_freq"])
        self.num_frames = int(train_cfg["num_frames"])
        self.max_grad_norm = float(train_cfg["max_grad_norm"])
        self.log_every = int(config["logging"]["log_every"])
        self.training_frame = 0

        set_seed(self.seed)
        try:
            from minatar import Environment
        except ImportError as exc:
            raise ImportError(
                "MinAtar is required to run this trainer. Please install it with "
                "`pip install -r requirements.txt`."
            ) from exc

        self.env = create_minatar_env(Environment, self.game, self.seed)
        self.obs_shape = self.env.state_shape()
        self.num_actions = self.env.num_actions()

        self.policy_net = MinAtarQNetwork(self.obs_shape[2], self.num_actions).to(self.device)
        self.target_net = MinAtarQNetwork(self.obs_shape[2], self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=float(train_cfg["learning_rate"]),
            alpha=float(train_cfg["rmsprop_alpha"]),
            centered=True,
            eps=float(train_cfg["rmsprop_eps"]),
        )

        use_das = self.algo == "fluid" and bool(fluid_cfg["use_das"])
        self.buffer = DensityAwarePrioritizedReplay(
            obs_shape=self.obs_shape,
            num_actions=self.num_actions,
            capacity=int(train_cfg["replay_capacity"]),
            priority_alpha=float(prio_cfg["alpha"]),
            priority_beta=float(prio_cfg["beta0"]),
            priority_eps=float(prio_cfg["eps"]),
            embedding_dim=int(train_cfg["hidden_dim"]),
            gamma=float(fluid_cfg["das_gamma"]),
            decay=float(fluid_cfg["das_decay"]),
            density_k=int(fluid_cfg["density_k"]),
            density_sigma=float(fluid_cfg["density_sigma"]),
            density_sample_size=int(fluid_cfg["density_sample_size"]),
            prune_candidates=int(fluid_cfg["prune_candidates"]),
            device=self.device,
            use_das=use_das,
            das_interval=int(fluid_cfg.get("das_interval", 1)),
            priority_max=float(prio_cfg.get("max_priority", 1e6)),
        )

        self.scl_eps = float(fluid_cfg["scl_eps"])
        self.scl_min_gate = float(fluid_cfg["scl_min_gate"])
        self.avgd_lambda = float(fluid_cfg["avgd_lambda"])
        self.policy_temperature = float(fluid_cfg["policy_temperature"])
        self.scl_weight_loss = bool(fluid_cfg.get("scl_weight_loss", False))
        self.phi_normalize = bool(fluid_cfg.get("phi_normalize", True))
        self.phi_priority_transform = str(fluid_cfg.get("phi_priority_transform", "log1p"))
        self.phi_priority_min = float(fluid_cfg.get("phi_priority_min", prio_cfg["eps"]))
        self.phi_priority_max = float(fluid_cfg.get("phi_priority_max", prio_cfg.get("max_priority", 10.0)))
        self.phi_scale = EmaScale(decay=float(fluid_cfg.get("phi_ema_decay", 0.99)))

        self.saber_cfg = config.get("saber", {})
        self.bridge_cfg = config.get("relo_bridge", {})
        self.relo_improve_cfg = config.get("relo_improve", {})
        self.component_ema_decay = float(
            self.saber_cfg.get(
                "ema_decay",
                self.bridge_cfg.get("ema_decay", self.relo_improve_cfg.get("ema_decay", 0.99)),
            )
        )
        self.component_priority_max = float(prio_cfg.get("max_priority", 10.0))
        self.component_scales: dict[str, EmaScale] = {}

        self.novelty_tracker = CountNoveltyTracker()
        self.variance_tracker = TargetVarianceTracker()

        run_dir = Path(exp_cfg["logdir"]) / self.game / self.algo / f"seed{self.seed}"
        self.logger = Logger(run_dir)
        self.output_path = run_dir / "result.pt"
        self.logger.info(f"Training {self.algo} on MinAtar/{self.game}")
        self.logger.info(f"Logging to {run_dir}")

    def _select_action(
        self, q_values: torch.Tensor, epsilon: float
    ) -> tuple[int, np.ndarray]:
        greedy_action = int(q_values.argmax(dim=1).item())
        behavior_probs = np.full(self.num_actions, epsilon / self.num_actions, dtype=np.float32)
        behavior_probs[greedy_action] += 1.0 - epsilon

        if random.random() < epsilon:
            action = random.randrange(self.num_actions)
        else:
            action = greedy_action
        return action, behavior_probs

    def _fluid_priority_from_phi(self, phi: torch.Tensor, update_scale: bool = True) -> torch.Tensor:
        priority = phi.detach().abs()
        if self.phi_normalize:
            priority = self.phi_scale.normalize(priority, update=update_scale)

        if self.phi_priority_transform == "log1p":
            priority = torch.log1p(priority)
        elif self.phi_priority_transform == "sqrt":
            priority = torch.sqrt(priority.clamp_min(0.0))
        elif self.phi_priority_transform == "none":
            pass
        else:
            raise ValueError(f"Unknown phi priority transform: {self.phi_priority_transform}")

        return priority.clamp(self.phi_priority_min, self.phi_priority_max)

    def _normalize_component(self, name: str, values: torch.Tensor) -> torch.Tensor:
        scale = self.component_scales.setdefault(name, EmaScale(self.component_ema_decay))
        normalized = scale.normalize(values.detach().abs(), update=True)
        return normalized.clamp(0.0, self.component_priority_max)

    def _schedule_value(self, start: float, end: float) -> float:
        horizon = max(int(self.config["training"]["eps_decay_frames"]), 1)
        progress = (self.training_frame - self.replay_start_size) / horizon
        progress = min(max(progress, 0.0), 1.0)
        return start + (end - start) * progress

    def _relo_signal(
        self,
        pre_loss: torch.Tensor,
        td_target: torch.Tensor,
        batch_states: torch.Tensor,
        batch_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            target_q = self.target_net(batch_states).gather(1, batch_actions)
            irr_loss = f.smooth_l1_loss(td_target, target_q, reduction="none").squeeze(-1)
            relo = torch.relu(pre_loss - irr_loss)
        return relo, irr_loss

    def _bellman_expectation_error(
        self,
        batch_states: torch.Tensor,
        batch_next_states: torch.Tensor,
        batch_rewards: torch.Tensor,
        batch_nonterminals: torch.Tensor,
        chosen_q: torch.Tensor,
    ) -> torch.Tensor:
        del batch_states
        with torch.no_grad():
            next_online_q = self.policy_net(batch_next_states)
            next_target_q = self.target_net(batch_next_states)
            probs = f.softmax(next_online_q / self.policy_temperature, dim=-1)
            expected_next_q = (probs * next_target_q).sum(dim=1, keepdim=True)
            expected_target = batch_rewards + batch_nonterminals * self.gamma * expected_next_q
            return (expected_target - chosen_q.detach()).abs().squeeze(-1)

    def _post_update_loss(self, batch: object, td_target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            post_q = self.policy_net(batch.states).gather(1, batch.actions)
            return f.smooth_l1_loss(td_target, post_q, reduction="none").squeeze(-1)

    def _saber_priority(
        self,
        relo: torch.Tensor,
        bellman_error: torch.Tensor,
        learning_progress: torch.Tensor,
        instability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        relo_n = self._normalize_component("saber_relo", relo)
        be_n = self._normalize_component("saber_be", bellman_error)
        lp_n = self._normalize_component("saber_lp", learning_progress)
        instab_n = self._normalize_component("saber_instab", instability)

        be_weight = self._schedule_value(
            float(self.saber_cfg.get("be_weight_start", 0.35)),
            float(self.saber_cfg.get("be_weight_end", 0.10)),
        )
        lp_weight = float(self.saber_cfg.get("progress_weight", 0.25))
        instab_weight = float(self.saber_cfg.get("instability_weight", 0.40))

        priority = relo_n + be_weight * be_n + lp_weight * lp_n - instab_weight * instab_n
        priority = priority.clamp(float(self.saber_cfg.get("priority_min", 1e-5)), self.component_priority_max)
        stats = {
            "saber_relo": float(relo_n.mean().cpu()),
            "saber_be": float(be_n.mean().cpu()),
            "saber_progress": float(lp_n.mean().cpu()),
            "saber_instability": float(instab_n.mean().cpu()),
            "saber_be_weight": float(be_weight),
            "saber_priority": float(priority.mean().cpu()),
        }
        return priority, stats

    def _relo_bridge_priority(
        self,
        relo: torch.Tensor,
        td_error: torch.Tensor,
        learning_progress: torch.Tensor,
        instability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        relo_n = self._normalize_component("bridge_relo", relo)
        td_n = self._normalize_component("bridge_td", td_error)
        lp_n = self._normalize_component("bridge_lp", learning_progress)
        instab_n = self._normalize_component("bridge_instab", instability)

        temp = max(float(self.bridge_cfg.get("confidence_temperature", 0.25)), 1e-6)
        confidence = torch.sigmoid((relo_n - float(self.bridge_cfg.get("confidence_threshold", 0.2))) / temp)
        hard_bridge = confidence * torch.log1p(td_n)

        bridge_weight = float(self.bridge_cfg.get("bridge_weight", 0.55))
        progress_weight = float(self.bridge_cfg.get("progress_weight", 0.20))
        instability_weight = float(self.bridge_cfg.get("instability_weight", 0.35))
        priority = relo_n + bridge_weight * hard_bridge + progress_weight * lp_n - instability_weight * instab_n
        priority = priority.clamp(float(self.bridge_cfg.get("priority_min", 1e-5)), self.component_priority_max)
        stats = {
            "bridge_relo": float(relo_n.mean().cpu()),
            "bridge_td": float(td_n.mean().cpu()),
            "bridge_confidence": float(confidence.mean().cpu()),
            "bridge_hard": float(hard_bridge.mean().cpu()),
            "bridge_progress": float(lp_n.mean().cpu()),
            "bridge_instability": float(instab_n.mean().cpu()),
            "bridge_priority": float(priority.mean().cpu()),
        }
        return priority, stats

    def _relo_mix_priority(
        self,
        relo: torch.Tensor,
        td_error: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        relo_n = self._normalize_component("relo_mix_relo", relo)
        td_n = self._normalize_component("relo_mix_td", td_error)
        td_signal = torch.log1p(td_n)
        td_weight = self._schedule_value(
            float(self.relo_improve_cfg.get("td_weight_start", 0.35)),
            float(self.relo_improve_cfg.get("td_weight_end", 0.10)),
        )

        priority = relo_n + td_weight * td_signal
        priority = priority.clamp(
            float(self.relo_improve_cfg.get("priority_min", 1e-5)),
            self.component_priority_max,
        )
        stats = {
            "relo_mix_relo": float(relo_n.mean().cpu()),
            "relo_mix_td": float(td_n.mean().cpu()),
            "relo_mix_td_weight": float(td_weight),
            "relo_mix_priority": float(priority.mean().cpu()),
        }
        return priority, stats

    def _relo_floor_priority(
        self,
        relo: torch.Tensor,
        td_error: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        relo_n = self._normalize_component("relo_floor_relo", relo)
        td_n = self._normalize_component("relo_floor_td", td_error)
        if "floor_weight_start" in self.relo_improve_cfg or "floor_weight_end" in self.relo_improve_cfg:
            floor_weight = self._schedule_value(
                float(self.relo_improve_cfg.get("floor_weight_start", 0.35)),
                float(self.relo_improve_cfg.get("floor_weight_end", 0.08)),
            )
        else:
            floor_weight = float(self.relo_improve_cfg.get("floor_weight", 0.35))
        floor = floor_weight * torch.log1p(td_n)

        priority = torch.maximum(relo_n, floor)
        priority = priority.clamp(
            float(self.relo_improve_cfg.get("priority_min", 1e-5)),
            self.component_priority_max,
        )
        stats = {
            "relo_floor_relo": float(relo_n.mean().cpu()),
            "relo_floor_td": float(td_n.mean().cpu()),
            "relo_floor_weight": float(floor_weight),
            "relo_floor_floor": float(floor.mean().cpu()),
            "relo_floor_priority": float(priority.mean().cpu()),
        }
        return priority, stats

    def _relo_adaptive_floor_priority(
        self,
        relo: torch.Tensor,
        td_error: torch.Tensor,
        learning_progress: torch.Tensor,
        instability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        relo_n = self._normalize_component("relo_adaptive_relo", relo)
        td_n = self._normalize_component("relo_adaptive_td", td_error)
        progress_n = self._normalize_component("relo_adaptive_progress", learning_progress)
        instability_n = self._normalize_component("relo_adaptive_instability", instability)

        td_signal = torch.log1p(td_n)
        eps = 1e-6
        coverage = relo_n / (relo_n + td_signal + eps)
        stability = progress_n / (progress_n + instability_n + eps)

        temp = max(float(self.relo_improve_cfg.get("adaptive_temperature", 0.15)), eps)
        coverage_threshold = float(self.relo_improve_cfg.get("coverage_threshold", 0.50))
        stability_threshold = float(self.relo_improve_cfg.get("stability_threshold", 0.50))
        floor_min = float(self.relo_improve_cfg.get("floor_weight_min", 0.02))
        floor_max = float(self.relo_improve_cfg.get("floor_weight_max", 0.35))

        coverage_gate = torch.sigmoid((coverage_threshold - coverage) / temp)
        stability_gate = torch.sigmoid((stability - stability_threshold) / temp)
        floor_weight = floor_min + (floor_max - floor_min) * coverage_gate * stability_gate
        floor = floor_weight * td_signal

        priority = torch.maximum(relo_n, floor)
        priority = priority.clamp(
            float(self.relo_improve_cfg.get("priority_min", 1e-5)),
            self.component_priority_max,
        )
        stats = {
            "adaptive_relo": float(relo_n.mean().cpu()),
            "adaptive_td": float(td_n.mean().cpu()),
            "adaptive_progress": float(progress_n.mean().cpu()),
            "adaptive_instability": float(instability_n.mean().cpu()),
            "adaptive_coverage": float(coverage.mean().cpu()),
            "adaptive_stability": float(stability.mean().cpu()),
            "adaptive_coverage_gate": float(coverage_gate.mean().cpu()),
            "adaptive_stability_gate": float(stability_gate.mean().cpu()),
            "adaptive_floor_weight": float(floor_weight.mean().cpu()),
            "adaptive_floor": float(floor.mean().cpu()),
            "adaptive_priority": float(priority.mean().cpu()),
        }
        return priority, stats

    def _relo_loss_factor(self, relo: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        relo_n = self._normalize_component("relo_wloss_relo", relo)
        loss_weight = float(self.relo_improve_cfg.get("loss_weight", 0.20))
        max_loss_factor = float(self.relo_improve_cfg.get("max_loss_factor", 1.50))
        factor = 1.0 + loss_weight * torch.log1p(relo_n)
        factor = factor.clamp(1.0, max_loss_factor)
        stats = {
            "relo_wloss_relo": float(relo_n.mean().cpu()),
            "relo_wloss_factor": float(factor.mean().cpu()),
        }
        return factor, stats

    def _priority_from_batch(
        self,
        chosen_q: torch.Tensor,
        td_target: torch.Tensor,
        batch_states: torch.Tensor,
        batch_actions: torch.Tensor,
        phi: torch.Tensor,
    ) -> np.ndarray:
        td_error = (td_target - chosen_q).abs().detach().squeeze(-1)
        if self.algo == "dqn":
            return np.ones(td_error.shape[0], dtype=np.float32)
        if self.algo == "per":
            return td_error.cpu().numpy()
        if self.algo == "relo":
            with torch.no_grad():
                target_q = self.target_net(batch_states).gather(1, batch_actions)
                orig_loss = f.smooth_l1_loss(td_target, chosen_q, reduction="none")
                irr_loss = f.smooth_l1_loss(td_target, target_q, reduction="none")
                relo = torch.relu(orig_loss - irr_loss).squeeze(-1)
            return relo.cpu().numpy()
        return phi.detach().cpu().numpy()

    def _soft_update(self) -> None:
        if self.target_tau >= 1.0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            return
        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.target_tau * param.data + (1.0 - self.target_tau) * target_param.data)

    def _collect_transition(self, frame: int, state: np.ndarray) -> tuple[np.ndarray, float, bool]:
        epsilon = epsilon_by_frame(
            frame,
            self.replay_start_size,
            self.config["training"]["eps_start"],
            self.config["training"]["eps_end"],
            self.config["training"]["eps_decay_frames"],
        )
        state_tensor = state_to_tensor(state, self.device)
        with torch.no_grad():
            q_values, features = self.policy_net(state_tensor, return_features=True)
        action, behavior_probs = self._select_action(q_values, epsilon)
        reward, terminated = self.env.act(action)
        next_state = self.env.state()
        next_state_tensor = state_to_tensor(next_state, self.device)

        with torch.no_grad():
            current_q = q_values[0, action]
            next_q = self.target_net(next_state_tensor).max(dim=1).values[0]
            td_target = reward + (0.0 if terminated else self.gamma * float(next_q.item()))
            td_error = torch.tensor(abs(td_target - float(current_q.item())), device=self.device)

        if self.algo == "fluid":
            novelty = self.novelty_tracker.observe(state)
            target_variance = self.variance_tracker.update(state, td_target)
            phi, _ = compute_phi(
                td_error.view(1),
                torch.tensor([target_variance], dtype=torch.float32, device=self.device),
                torch.tensor([novelty], dtype=torch.float32, device=self.device),
                self.scl_eps,
                self.scl_min_gate,
            )
            learnability = float(self._fluid_priority_from_phi(phi, update_scale=False).item())
        else:
            novelty = 1.0
            target_variance = 0.0
            learnability = float(td_error.item())

        self.buffer.add(
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            nonterminal=0.0 if terminated else 1.0,
            behavior_probs=behavior_probs,
            novelty=float(novelty),
            target_variance=float(target_variance),
            learnability=max(learnability, 1e-6),
            embedding=features.squeeze(0).detach().cpu().numpy(),
        )
        return next_state, float(reward), bool(terminated)

    def _train_step(self) -> dict[str, float]:
        batch = self.buffer.sample(self.batch_size)

        q_values = self.policy_net(batch.states)
        chosen_q = q_values.gather(1, batch.actions)
        with torch.no_grad():
            next_q = self.target_net(batch.next_states).max(dim=1, keepdim=True).values
            td_target = batch.rewards + batch.nonterminals * self.gamma * next_q

        td_error = (td_target - chosen_q).abs().squeeze(-1)
        base_loss = f.smooth_l1_loss(td_target, chosen_q, reduction="none")
        pre_loss = base_loss.detach().squeeze(-1)
        loss_stats = {}

        if self.algo == "fluid":
            phi, scl_factor = compute_phi(
                td_error,
                batch.target_variance,
                batch.novelty,
                self.scl_eps,
                self.scl_min_gate,
            )
            loss_factor = scl_factor if self.scl_weight_loss else torch.ones_like(scl_factor)
            weighted_loss = batch.is_weights * base_loss * loss_factor.unsqueeze(-1)
            avgd_penalty, avgd_stats = compute_avgd_penalty(
                q_values, batch.behavior_probs, self.policy_temperature
            )
            avgd_loss = self.avgd_lambda * avgd_penalty.mean()
        elif self.algo in {"relo_wloss", "relo_floor_wloss", "blcr"}:
            phi = td_error
            loss_relo, _ = self._relo_signal(pre_loss, td_target.detach(), batch.states, batch.actions)
            loss_factor, loss_stats = self._relo_loss_factor(loss_relo)
            weighted_loss = batch.is_weights * base_loss * loss_factor.unsqueeze(-1)
            avgd_stats = {}
            avgd_loss = torch.zeros((), device=self.device)
        else:
            phi = td_error
            weighted_loss = batch.is_weights * base_loss
            avgd_stats = {}
            avgd_loss = torch.zeros((), device=self.device)

        loss = weighted_loss.mean() + avgd_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.algo == "fluid":
            phi_priority = self._fluid_priority_from_phi(phi, update_scale=True)
            priorities = phi_priority.cpu().numpy()
            learnability = priorities
            loss_values = None
            custom_stats = {}
        elif self.algo in {
            "saber",
            "relo_bridge",
            "relo_mix",
            "relo_floor",
            "relo_floor_adaptive",
            "relo_floor_decay",
            "relo_wloss",
            "relo_floor_wloss",
            "blcr",
        }:
            post_loss = self._post_update_loss(batch, td_target)
            learning_progress = torch.relu(pre_loss - post_loss)
            instability = torch.relu(post_loss - pre_loss)
            relo, irr_loss = self._relo_signal(pre_loss, td_target.detach(), batch.states, batch.actions)
            if self.algo == "saber":
                bellman_error = self._bellman_expectation_error(
                    batch.states,
                    batch.next_states,
                    batch.rewards,
                    batch.nonterminals,
                    chosen_q.detach(),
                )
                priority_tensor, custom_stats = self._saber_priority(
                    relo, bellman_error, learning_progress, instability
                )
                custom_stats["saber_bellman_error"] = float(bellman_error.mean().cpu())
            elif self.algo == "relo_bridge":
                priority_tensor, custom_stats = self._relo_bridge_priority(
                    relo, td_error.detach(), learning_progress, instability
                )
            elif self.algo == "relo_mix":
                priority_tensor, custom_stats = self._relo_mix_priority(relo, td_error.detach())
            elif self.algo == "relo_floor_adaptive":
                priority_tensor, custom_stats = self._relo_adaptive_floor_priority(
                    relo, td_error.detach(), learning_progress, instability
                )
            elif self.algo in {"relo_floor", "relo_floor_decay", "relo_floor_wloss", "blcr"}:
                priority_tensor, custom_stats = self._relo_floor_priority(relo, td_error.detach())
                custom_stats.update(loss_stats)
            else:
                priority_tensor = relo.detach().clamp(
                    float(self.relo_improve_cfg.get("priority_min", 1e-5)),
                    self.component_priority_max,
                )
                custom_stats = loss_stats
            custom_stats["relo_raw"] = float(relo.mean().cpu())
            custom_stats["irr_loss"] = float(irr_loss.mean().cpu())
            custom_stats["post_loss"] = float(post_loss.mean().cpu())
            priorities = priority_tensor.cpu().numpy()
            learnability = priorities
            loss_values = post_loss.cpu().numpy()
            phi_priority = priority_tensor
        else:
            phi_priority = phi
            priorities = self._priority_from_batch(
                chosen_q.detach(), td_target.detach(), batch.states, batch.actions, phi
            )
            learnability = None
            loss_values = None
            custom_stats = {}
        self.buffer.update_priorities(batch.data_indices, priorities, learnability, loss_values)

        metrics = {
            "loss": float(loss.detach().cpu()),
            "td_error": float(td_error.mean().detach().cpu()),
            "priority_beta": float(self.buffer.priority_beta),
        }
        metrics.update(custom_stats)
        if self.algo == "fluid":
            metrics["phi"] = float(phi.mean().detach().cpu())
            metrics["phi_priority"] = float(phi_priority.mean().detach().cpu())
            metrics["phi_scale"] = float(self.phi_scale.scale)
            metrics["scl_factor"] = float(scl_factor.mean().detach().cpu())
            metrics.update(avgd_stats)
        return metrics

    def train(self) -> None:
        train_cfg = self.config["training"]
        priority_beta_increase = (
            (1.0 - self.buffer.priority_beta)
            / max(self.num_frames - self.replay_start_size, 1)
        )

        frame = 0
        episode = 0
        returns: list[float] = []
        frame_stamps: list[int] = []
        moving_return = 0.0
        last_metrics: dict[str, float] = {}
        start_time = time.time()
        show_progress = bool(train_cfg.get("show_progress", False))
        progress = tqdm(total=self.num_frames) if show_progress else None
        log_interval = max(self.log_every, 1)
        next_log_frame = log_interval

        while frame < self.num_frames:
            self.env.reset()
            state = self.env.state()
            done = False
            episode_return = 0.0

            while not done and frame < self.num_frames:
                next_state, reward, done = self._collect_transition(frame, state)
                frame += 1
                if progress is not None:
                    progress.update(1)
                episode_return += reward
                state = next_state

                if frame > self.replay_start_size and len(self.buffer) >= self.batch_size:
                    self.buffer.anneal_beta(self.buffer.priority_beta + priority_beta_increase)
                    if frame % self.train_freq == 0:
                        self.training_frame = frame
                        last_metrics = self._train_step()
                    if frame % self.target_update_freq == 0:
                        self._soft_update()

            episode += 1
            returns.append(episode_return)
            frame_stamps.append(frame)
            moving_return = 0.99 * moving_return + 0.01 * episode_return

            if frame >= next_log_frame or frame >= self.num_frames:
                self.logger.step = frame
                self.logger.scalar("return", episode_return)
                self.logger.scalar("avg_return", moving_return)
                self.logger.scalar("episode", episode)
                self.logger.scalar("buffer_size", len(self.buffer))
                for key, value in last_metrics.items():
                    self.logger.scalar(key, value)
                self.logger.write(with_fps=True)
                while next_log_frame <= frame:
                    next_log_frame += log_interval

            if episode % int(train_cfg["console_log_every"]) == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                self.logger.info(
                    f"Episode={episode} Frame={frame} Return={episode_return:.2f} "
                    f"AvgReturn={moving_return:.2f} FPS={frame / elapsed:.1f}"
                )

        if progress is not None:
            progress.close()
        torch.save(
            {
                "returns": returns,
                "frame_stamps": frame_stamps,
                "policy_net_state_dict": self.policy_net.state_dict(),
                "config": self.config,
            },
            self.output_path,
        )
        self.logger.info(f"Saved results to {self.output_path}")
        self.logger.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--game", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--algo", type=str)
    parser.add_argument("--override", action="append", default=[])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.game:
        config["experiment"]["game"] = args.game
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.algo:
        config["experiment"]["algo"] = args.algo

    for raw in args.override:
        key, value = parse_override(raw)
        set_by_dotted_key(config, key, value)

    os.makedirs(config["experiment"]["logdir"], exist_ok=True)
    trainer = DQNTrainer(config)
    trainer.train()

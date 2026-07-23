from __future__ import annotations

import argparse
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as f
import torch.optim as optim

from fluid.config import load_config, parse_override, set_by_dotted_key
from fluid.envs import make_external_discrete_env
from fluid.models.vector_q_network import VectorQNetwork
from fluid.replay.buffer import DensityAwarePrioritizedReplay
from fluid.utils.logging import Logger
from fluid.utils.seed import set_seed


def choose_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug or "external_env"


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


class ExternalDiscreteTrainer:
    """DQN/ReLo trainer for vector observations and discrete action spaces.

    This trainer is intentionally separate from the MinAtar trainer so optional
    ALE/DMControl probes cannot alter the reproduced MinAtar baselines.
    """

    def __init__(self, config: dict):
        self.config = config
        exp_cfg = config["experiment"]
        env_cfg = config["env"]
        train_cfg = config["training"]
        prio_cfg = config["prioritization"]
        fluid_cfg = config.get("fluid", {})

        self.algo = str(exp_cfg["algo"]).lower()
        self.seed = int(exp_cfg["seed"])
        self.device = choose_device(str(config["system"]["device"]))
        self.gamma = float(train_cfg["gamma"])
        self.batch_size = int(train_cfg["batch_size"])
        self.replay_start_size = int(train_cfg["replay_start_size"])
        self.target_update_freq = int(train_cfg["target_update_freq"])
        self.target_tau = float(train_cfg["target_tau"])
        self.train_freq = int(train_cfg["train_freq"])
        self.num_frames = int(train_cfg["num_frames"])
        self.max_episodes = int(train_cfg.get("max_episodes", 0))
        self.max_grad_norm = float(train_cfg["max_grad_norm"])
        self.max_episode_steps = int(train_cfg.get("max_episode_steps", 0))
        self.reward_clip = float(train_cfg.get("reward_clip", 0.0))
        self.log_every = int(config["logging"]["log_every"])
        self.training_frame = 0

        set_seed(self.seed)
        self.env = make_external_discrete_env(config, self.seed)
        self.obs_scale = float(env_cfg.get("obs_scale", 1.0))
        self._initial_state = self._preprocess_obs(self.env.reset())
        self.obs_shape = (int(self._initial_state.size),)
        self.num_actions = int(self.env.num_actions)

        hidden_dim = int(train_cfg.get("hidden_dim", 256))
        hidden_layers = int(train_cfg.get("hidden_layers", 2))
        embedding_dim = hidden_dim if hidden_layers > 0 else self.obs_shape[0]
        self.policy_net = VectorQNetwork(
            input_dim=self.obs_shape[0],
            num_actions=self.num_actions,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        ).to(self.device)
        self.target_net = VectorQNetwork(
            input_dim=self.obs_shape[0],
            num_actions=self.num_actions,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        ).to(self.device)
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

        self.buffer = DensityAwarePrioritizedReplay(
            obs_shape=self.obs_shape,
            num_actions=self.num_actions,
            capacity=int(train_cfg["replay_capacity"]),
            priority_alpha=float(prio_cfg["alpha"]),
            priority_beta=float(prio_cfg["beta0"]),
            priority_eps=float(prio_cfg["eps"]),
            embedding_dim=embedding_dim,
            gamma=float(fluid_cfg.get("das_gamma", 1.0)),
            decay=float(fluid_cfg.get("das_decay", 0.0)),
            density_k=int(fluid_cfg.get("density_k", 8)),
            density_sigma=float(fluid_cfg.get("density_sigma", 1.0)),
            density_sample_size=int(fluid_cfg.get("density_sample_size", 64)),
            prune_candidates=int(fluid_cfg.get("prune_candidates", 64)),
            device=self.device,
            use_das=False,
            das_interval=int(fluid_cfg.get("das_interval", 1)),
            priority_max=float(prio_cfg.get("max_priority", 10.0)),
        )

        self.relo_improve_cfg = config.get("relo_improve", {})
        self.component_ema_decay = float(self.relo_improve_cfg.get("ema_decay", 0.99))
        self.component_priority_max = float(prio_cfg.get("max_priority", 10.0))
        self.component_scales: dict[str, EmaScale] = {}

        self.env_name = self._resolve_env_name()
        run_dir = Path(exp_cfg["logdir"]) / self.env_name / self.algo / f"seed{self.seed}"
        self.logger = Logger(run_dir)
        self.output_path = run_dir / "result.pt"
        self.logger.info(f"Training {self.algo} on {self.env_name}")
        self.logger.info(f"ObsDim={self.obs_shape[0]} Actions={self.num_actions} Device={self.device}")
        self.logger.info(f"Logging to {run_dir}")

    def _resolve_env_name(self) -> str:
        exp_cfg = self.config["experiment"]
        env_cfg = self.config["env"]
        if exp_cfg.get("env_name"):
            return slugify(str(exp_cfg["env_name"]))
        env_type = str(env_cfg["type"]).lower()
        if env_type in {"gym", "ale"}:
            return slugify(str(env_cfg["id"]))
        return slugify(f"{env_cfg.get('domain', env_type)}_{env_cfg.get('task', 'task')}")

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        state = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.obs_scale != 1.0:
            state = state / self.obs_scale
        return state.astype(np.float32, copy=False)

    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)

    def _select_action(self, q_values: torch.Tensor, epsilon: float) -> tuple[int, np.ndarray]:
        greedy_action = int(q_values.argmax(dim=1).item())
        behavior_probs = np.full(self.num_actions, epsilon / self.num_actions, dtype=np.float32)
        behavior_probs[greedy_action] += 1.0 - epsilon

        if random.random() < epsilon:
            action = random.randrange(self.num_actions)
        else:
            action = greedy_action
        return action, behavior_probs

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

    def _post_update_loss(self, batch: object, td_target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            post_q = self.policy_net(batch.states).gather(1, batch.actions)
            return f.smooth_l1_loss(td_target, post_q, reduction="none").squeeze(-1)

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
        loss_weight = float(self.relo_improve_cfg.get("loss_weight", 0.08))
        max_loss_factor = float(self.relo_improve_cfg.get("max_loss_factor", 1.25))
        factor = 1.0 + loss_weight * torch.log1p(relo_n)
        factor = factor.clamp(1.0, max_loss_factor)
        stats = {
            "relo_wloss_relo": float(relo_n.mean().cpu()),
            "relo_wloss_factor": float(factor.mean().cpu()),
        }
        return factor, stats

    def _soft_update(self) -> None:
        if self.target_tau >= 1.0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            return
        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.target_tau * param.data + (1.0 - self.target_tau) * target_param.data)

    def _clip_reward(self, reward: float) -> float:
        if self.reward_clip <= 0.0:
            return reward
        return float(np.clip(reward, -self.reward_clip, self.reward_clip))

    def _collect_transition(
        self, frame: int, state: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool]:
        epsilon = epsilon_by_frame(
            frame,
            self.replay_start_size,
            self.config["training"]["eps_start"],
            self.config["training"]["eps_end"],
            self.config["training"]["eps_decay_frames"],
        )
        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values, features = self.policy_net(state_tensor, return_features=True)
        action, behavior_probs = self._select_action(q_values, epsilon)
        step_result = self.env.step(action)
        next_state = self._preprocess_obs(step_result.observation)
        raw_reward = float(step_result.reward)
        train_reward = self._clip_reward(raw_reward)
        terminated = bool(step_result.terminated)

        next_state_tensor = self._state_to_tensor(next_state)
        with torch.no_grad():
            current_q = q_values[0, action]
            next_q = self.target_net(next_state_tensor).max(dim=1).values[0]
            td_target = train_reward + (0.0 if terminated else self.gamma * float(next_q.item()))
            td_error = abs(td_target - float(current_q.item()))

        self.buffer.add(
            state=state,
            action=action,
            reward=train_reward,
            next_state=next_state,
            nonterminal=0.0 if terminated else 1.0,
            behavior_probs=behavior_probs,
            novelty=1.0,
            target_variance=0.0,
            learnability=max(float(td_error), 1e-6),
            embedding=features.squeeze(0).detach().cpu().numpy(),
        )
        return next_state, raw_reward, terminated, bool(step_result.truncated)

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
        loss_stats: dict[str, float] = {}

        if self.algo in {"relo_wloss", "relo_floor_wloss", "blcr"}:
            loss_relo, _ = self._relo_signal(pre_loss, td_target.detach(), batch.states, batch.actions)
            loss_factor, loss_stats = self._relo_loss_factor(loss_relo)
            weighted_loss = batch.is_weights * base_loss * loss_factor.unsqueeze(-1)
        else:
            weighted_loss = batch.is_weights * base_loss

        loss = weighted_loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        custom_stats: dict[str, float] = {}
        loss_values = None
        learnability = None
        if self.algo == "dqn":
            priorities = np.ones(td_error.shape[0], dtype=np.float32)
        elif self.algo == "per":
            priorities = td_error.detach().cpu().numpy()
        elif self.algo in {"relo", "relo_wloss"}:
            post_loss = self._post_update_loss(batch, td_target)
            relo, irr_loss = self._relo_signal(pre_loss, td_target.detach(), batch.states, batch.actions)
            priority_tensor = relo.detach().clamp(
                float(self.relo_improve_cfg.get("priority_min", 1e-5)),
                self.component_priority_max,
            )
            custom_stats.update(loss_stats)
            custom_stats["relo_raw"] = float(relo.mean().cpu())
            custom_stats["irr_loss"] = float(irr_loss.mean().cpu())
            custom_stats["post_loss"] = float(post_loss.mean().cpu())
            priorities = priority_tensor.cpu().numpy()
            learnability = priorities
            loss_values = post_loss.cpu().numpy()
        elif self.algo in {
            "relo_floor",
            "relo_floor_adaptive",
            "relo_floor_decay",
            "relo_floor_wloss",
            "blcr",
        }:
            post_loss = self._post_update_loss(batch, td_target)
            relo, irr_loss = self._relo_signal(pre_loss, td_target.detach(), batch.states, batch.actions)
            learning_progress = torch.relu(pre_loss - post_loss)
            instability = torch.relu(post_loss - pre_loss)
            if self.algo == "relo_floor_adaptive":
                priority_tensor, custom_stats = self._relo_adaptive_floor_priority(
                    relo, td_error.detach(), learning_progress, instability
                )
            else:
                priority_tensor, custom_stats = self._relo_floor_priority(relo, td_error.detach())
            custom_stats.update(loss_stats)
            custom_stats["relo_raw"] = float(relo.mean().cpu())
            custom_stats["irr_loss"] = float(irr_loss.mean().cpu())
            custom_stats["post_loss"] = float(post_loss.mean().cpu())
            priorities = priority_tensor.cpu().numpy()
            learnability = priorities
            loss_values = post_loss.cpu().numpy()
        else:
            raise ValueError(f"Unsupported algorithm for external discrete trainer: {self.algo}")

        self.buffer.update_priorities(batch.data_indices, priorities, learnability, loss_values)

        metrics = {
            "loss": float(loss.detach().cpu()),
            "td_error": float(td_error.mean().detach().cpu()),
            "priority_beta": float(self.buffer.priority_beta),
        }
        metrics.update(custom_stats)
        return metrics

    def train(self) -> None:
        train_cfg = self.config["training"]
        priority_beta_increase = (
            (1.0 - self.buffer.priority_beta)
            / max(self.num_frames - self.replay_start_size, 1)
        )

        frame = 0
        episode = 0
        episode_steps = 0
        episode_return = 0.0
        moving_return = 0.0
        returns: list[float] = []
        frame_stamps: list[int] = []
        last_completed_return: float | None = None
        last_metrics: dict[str, float] = {}
        start_time = time.time()
        log_interval = max(self.log_every, 1)
        next_log_frame = log_interval
        state = self._initial_state

        while frame < self.num_frames and (self.max_episodes <= 0 or episode < self.max_episodes):
            next_state, reward, terminated, truncated = self._collect_transition(frame, state)
            frame += 1
            episode_steps += 1
            episode_return += reward

            if frame > self.replay_start_size and len(self.buffer) >= self.batch_size:
                self.buffer.anneal_beta(self.buffer.priority_beta + priority_beta_increase)
                if frame % self.train_freq == 0:
                    self.training_frame = frame
                    last_metrics = self._train_step()
                if frame % self.target_update_freq == 0:
                    self._soft_update()

            capped = self.max_episode_steps > 0 and episode_steps >= self.max_episode_steps
            done = terminated or truncated or capped
            if done:
                episode += 1
                returns.append(episode_return)
                frame_stamps.append(frame)
                moving_return = 0.99 * moving_return + 0.01 * episode_return
                last_completed_return = episode_return
                if frame < self.num_frames and (self.max_episodes <= 0 or episode < self.max_episodes):
                    state = self._preprocess_obs(self.env.reset())
                    episode_steps = 0
                    episode_return = 0.0
                else:
                    state = next_state
            else:
                state = next_state

            if frame >= next_log_frame or frame >= self.num_frames:
                self.logger.step = frame
                logged_return = episode_return if last_completed_return is None else last_completed_return
                self.logger.scalar("return", logged_return)
                self.logger.scalar("current_episode_return", episode_return)
                self.logger.scalar("avg_return", moving_return)
                self.logger.scalar("episode", episode)
                self.logger.scalar("episode_steps", episode_steps)
                self.logger.scalar("buffer_size", len(self.buffer))
                for key, value in last_metrics.items():
                    self.logger.scalar(key, value)
                self.logger.write(with_fps=True)
                while next_log_frame <= frame:
                    next_log_frame += log_interval

            console_log_every = int(train_cfg.get("console_log_every", 0))
            if done and console_log_every > 0 and episode % console_log_every == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                self.logger.info(
                    f"Episode={episode} Frame={frame} Return={last_completed_return:.2f} "
                    f"AvgReturn={moving_return:.2f} FPS={frame / elapsed:.1f}"
                )

        if episode_steps > 0 and (not returns or frame_stamps[-1] != frame):
            returns.append(episode_return)
            frame_stamps.append(frame)

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
        self.env.close()
        self.logger.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--algo", type=str)
    parser.add_argument("--env-name", type=str)
    parser.add_argument("--override", action="append", default=[])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.algo:
        config["experiment"]["algo"] = args.algo
    if args.env_name:
        config["experiment"]["env_name"] = args.env_name

    for raw in args.override:
        key, value = parse_override(raw)
        set_by_dotted_key(config, key, value)

    os.makedirs(config["experiment"]["logdir"], exist_ok=True)
    trainer = ExternalDiscreteTrainer(config)
    trainer.train()

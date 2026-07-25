from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


METHODS = {"sac", "per_sac", "relo_sac", "blcr_sac"}


@dataclass
class TrainConfig:
    env: str
    method: str
    seed: int = 0
    total_steps: int = 300_000
    learning_starts: int = 5_000
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_size: int = 256
    hidden_layers: int = 2
    update_every: int = 1
    gradient_steps: int = 1
    eval_every: int = 10_000
    eval_episodes: int = 5
    max_episode_steps: int = 1000
    device: str = "auto"
    logdir: str = "logs_blcr_sac"
    per_alpha: float = 0.6
    beta_start: float = 0.4
    priority_eps: float = 1e-6
    blcr_floor_lambda: float = 0.35
    blcr_loss_eta: float = 0.08
    blcr_loss_wmax: float = 1.5
    signal_clip: float = 10.0
    autotune_entropy: bool = True
    fixed_entropy_alpha: float = 0.2
    num_threads: int = 1


def flatten_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in sorted(obs)]).astype(np.float32)
    if isinstance(obs, (tuple, list)):
        return np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in obs]).astype(np.float32)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


class DMCGymWrapper:
    def __init__(self, domain: str, task: str, seed: int, max_episode_steps: int):
        from dm_control import suite
        from gymnasium import spaces

        self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={"random": seed})
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        action_spec = self._env.action_spec()
        self.action_space = spaces.Box(
            low=np.asarray(action_spec.minimum, dtype=np.float32),
            high=np.asarray(action_spec.maximum, dtype=np.float32),
            dtype=np.float32,
        )
        obs = flatten_obs(self._env.reset().observation)
        high = np.full(obs.shape, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self, seed: int | None = None):
        self._elapsed_steps = 0
        timestep = self._env.reset()
        return flatten_obs(timestep.observation), {}

    def step(self, action: np.ndarray):
        timestep = self._env.step(action)
        self._elapsed_steps += 1
        reward = 0.0 if timestep.reward is None else float(timestep.reward)
        terminated = bool(timestep.last())
        truncated = self._elapsed_steps >= self._max_episode_steps
        return flatten_obs(timestep.observation), reward, terminated, truncated, {}

    def close(self):
        return None


def make_env(env_id: str, seed: int, max_episode_steps: int):
    if env_id.startswith("dmcontrol_"):
        parts = env_id.split("_")
        return DMCGymWrapper(parts[1], "_".join(parts[2:]), seed, max_episode_steps)
    if env_id.startswith(("Fetch", "Adroit", "Hand")):
        import gymnasium_robotics  # noqa: F401
    if env_id.startswith(("Cont-", "Finite-")):
        import gym_electric_motor  # noqa: F401

    env = gym.make(env_id)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass
    return env


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, seed: int, priority_eps: float):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.max_priority = 1.0
        self.priority_eps = priority_eps
        self.ptr = 0
        self.size = 0
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, done):
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = done
        self.priorities[idx] = max(self.max_priority, self.priority_eps)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, method: str, alpha: float, beta: float, device: torch.device):
        if method == "sac":
            idxs = self.rng.integers(0, self.size, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            raw = np.maximum(self.priorities[: self.size].astype(np.float64), self.priority_eps)
            probs = raw**alpha
            probs = probs / probs.sum() if np.isfinite(probs.sum()) and probs.sum() > 0 else np.full(self.size, 1.0 / self.size)
            idxs = self.rng.choice(self.size, size=batch_size, replace=True, p=probs)
            weights = (self.size * probs[idxs]) ** (-beta)
            weights = (weights / (weights.max() + 1e-8)).astype(np.float32)
        return {
            "obs": torch.as_tensor(self.obs[idxs], device=device),
            "actions": torch.as_tensor(self.actions[idxs], device=device),
            "rewards": torch.as_tensor(self.rewards[idxs], device=device).unsqueeze(-1),
            "next_obs": torch.as_tensor(self.next_obs[idxs], device=device),
            "dones": torch.as_tensor(self.dones[idxs], device=device).unsqueeze(-1),
            "weights": torch.as_tensor(weights, device=device).unsqueeze(-1),
            "idxs": idxs,
        }

    def update_priorities(self, idxs: np.ndarray, priorities: np.ndarray):
        priorities = np.maximum(np.asarray(priorities, dtype=np.float32).reshape(-1), self.priority_eps)
        self.priorities[idxs] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max(initial=self.priority_eps)))


def mlp(input_dim: int, output_dim: int, hidden_size: int, hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for _ in range(hidden_layers):
        layers += [nn.Linear(last, hidden_size), nn.ReLU()]
        last = hidden_size
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int, hidden_layers: int):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, hidden_size, hidden_layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_space: gym.spaces.Box, hidden_size: int, hidden_layers: int):
        super().__init__()
        act_dim = int(np.prod(action_space.shape))
        self.backbone = mlp(obs_dim, hidden_size, hidden_size, hidden_layers)
        self.mu = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Linear(hidden_size, act_dim)
        scale = (action_space.high - action_space.low) / 2.0
        bias = (action_space.high + action_space.low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        return self.mu(h), torch.clamp(self.log_std(h), -5.0, 2.0)

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        mu, log_std = self(obs)
        normal = Normal(mu, log_std.exp())
        x_t = mu if deterministic else normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


def normalized_signal(x: torch.Tensor, clip: float) -> torch.Tensor:
    x = torch.clamp(x.detach(), min=0.0)
    return torch.clamp(x / x.mean().clamp_min(1e-6), 0.0, clip)


def evaluate(actor: SquashedGaussianActor, cfg: TrainConfig, device: torch.device):
    env = make_env(cfg.env, cfg.seed + 100_000, cfg.max_episode_steps)
    returns, lengths = [], []
    for ep in range(cfg.eval_episodes):
        obs_raw, _ = env.reset(seed=cfg.seed + 100_000 + ep)
        obs = flatten_obs(obs_raw)
        ep_ret, ep_len, done = 0.0, 0, False
        while not done and ep_len < cfg.max_episode_steps:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.sample(obs_t, deterministic=True)
            next_obs_raw, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            obs = flatten_obs(next_obs_raw)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
            ep_len += 1
        returns.append(ep_ret)
        lengths.append(ep_len)
    env.close()
    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(lengths))


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, p_target in zip(source.parameters(), target.parameters()):
            p_target.data.mul_(1.0 - tau).add_(tau * p.data)


def update(cfg: TrainConfig, step: int, replay: ReplayBuffer, actor, q1, q2, q1_t, q2_t, actor_opt, q_opt, log_alpha, alpha_opt, target_entropy, device):
    beta = min(1.0, cfg.beta_start + step * (1.0 - cfg.beta_start) / max(1, cfg.total_steps))
    batch = replay.sample(cfg.batch_size, cfg.method, cfg.per_alpha, beta, device)
    obs, actions = batch["obs"], batch["actions"]
    rewards, next_obs, dones = batch["rewards"], batch["next_obs"], batch["dones"]
    alpha = log_alpha.exp().detach() if cfg.autotune_entropy else torch.tensor(cfg.fixed_entropy_alpha, device=device)

    with torch.no_grad():
        next_actions, next_logp = actor.sample(next_obs)
        next_q = torch.min(q1_t(next_obs, next_actions), q2_t(next_obs, next_actions))
        target_q = rewards + cfg.gamma * (1.0 - dones) * (next_q - alpha * next_logp)

    q1_pred, q2_pred = q1(obs, actions), q2(obs, actions)
    pre_loss = 0.5 * ((q1_pred - target_q).pow(2) + (q2_pred - target_q).pow(2))
    td_abs = 0.5 * ((q1_pred - target_q).abs() + (q2_pred - target_q).abs())
    loss_weights = torch.ones_like(pre_loss)
    if cfg.method == "blcr_sac":
        td_norm = normalized_signal(td_abs, cfg.signal_clip)
        loss_weights = torch.clamp(
            1.0 + cfg.blcr_loss_eta * torch.log1p(td_norm),
            1.0,
            cfg.blcr_loss_wmax,
        )

    q_loss = (batch["weights"] * loss_weights * pre_loss).mean()
    q_opt.zero_grad(set_to_none=True)
    q_loss.backward()
    q_opt.step()

    with torch.no_grad():
        q1_post, q2_post = q1(obs, actions), q2(obs, actions)
        post_loss = 0.5 * ((q1_post - target_q).pow(2) + (q2_post - target_q).pow(2))
        reducible = torch.clamp(pre_loss - post_loss, min=0.0)
        post_td_abs = 0.5 * ((q1_post - target_q).abs() + (q2_post - target_q).abs())
        if cfg.method == "per_sac":
            priority = post_td_abs + cfg.priority_eps
        elif cfg.method == "relo_sac":
            priority = normalized_signal(reducible, cfg.signal_clip) + cfg.priority_eps
        elif cfg.method == "blcr_sac":
            red_norm = normalized_signal(reducible, cfg.signal_clip)
            floor = cfg.blcr_floor_lambda * torch.log1p(
                normalized_signal(post_td_abs, cfg.signal_clip)
            )
            priority = torch.maximum(red_norm, floor) + cfg.priority_eps
        else:
            priority = None
        if priority is not None:
            replay.update_priorities(batch["idxs"], priority.squeeze(-1).cpu().numpy())

    pi, logp = actor.sample(obs)
    actor_loss = (alpha * logp - torch.min(q1(obs, pi), q2(obs, pi))).mean()
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_opt.step()

    alpha_loss_value = 0.0
    if cfg.autotune_entropy and alpha_opt is not None:
        alpha_loss = -(log_alpha.exp() * (logp + target_entropy).detach()).mean()
        alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_opt.step()
        alpha_loss_value = float(alpha_loss.detach().cpu())

    soft_update(q1, q1_t, cfg.tau)
    soft_update(q2, q2_t, cfg.tau)
    return {
        "q_loss": float(q_loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
        "alpha": float(log_alpha.exp().detach().cpu()) if cfg.autotune_entropy else cfg.fixed_entropy_alpha,
        "alpha_loss": alpha_loss_value,
        "td_abs": float(td_abs.mean().detach().cpu()),
        "reducible": float(reducible.mean().detach().cpu()),
        "priority_mean": float(np.mean(replay.priorities[: replay.size])) if replay.size else 0.0,
    }


def train(cfg: TrainConfig) -> Path:
    if cfg.method not in METHODS:
        raise ValueError(f"Unknown method {cfg.method}")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, cfg.num_threads))
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device if cfg.device != "auto" else "cpu")

    env = make_env(cfg.env, cfg.seed, cfg.max_episode_steps)
    obs_raw, _ = env.reset(seed=cfg.seed)
    obs = flatten_obs(obs_raw)
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError(f"BLCR-SAC requires Box actions, got {env.action_space}")
    obs_dim, act_dim = int(obs.shape[0]), int(np.prod(env.action_space.shape))

    run_dir = Path(cfg.logdir) / cfg.method / cfg.env.replace("/", "_") / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    metrics_f = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
    writer = SummaryWriter(str(run_dir))

    replay = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size, cfg.seed, cfg.priority_eps)
    actor = SquashedGaussianActor(obs_dim, env.action_space, cfg.hidden_size, cfg.hidden_layers).to(device)
    q1 = SoftQNetwork(obs_dim, act_dim, cfg.hidden_size, cfg.hidden_layers).to(device)
    q2 = SoftQNetwork(obs_dim, act_dim, cfg.hidden_size, cfg.hidden_layers).to(device)
    q1_t = SoftQNetwork(obs_dim, act_dim, cfg.hidden_size, cfg.hidden_layers).to(device)
    q2_t = SoftQNetwork(obs_dim, act_dim, cfg.hidden_size, cfg.hidden_layers).to(device)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.policy_lr)
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.q_lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr) if cfg.autotune_entropy else None
    target_entropy = -float(act_dim)

    ep_ret, ep_len, episode = 0.0, 0, 0
    start = time.time()
    last_update: dict[str, float] = {}
    for step in range(1, cfg.total_steps + 1):
        if step < cfg.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _ = actor.sample(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            action = action.cpu().numpy()[0]

        next_obs_raw, reward, terminated, truncated, _ = env.step(action)
        next_obs = flatten_obs(next_obs_raw)
        done = bool(terminated or truncated)
        replay.add(obs, action, float(reward), next_obs, float(terminated))
        obs = next_obs
        ep_ret += float(reward)
        ep_len += 1

        if done or ep_len >= cfg.max_episode_steps:
            writer.add_scalar("train/episode_return", ep_ret, step)
            metrics_f.write(json.dumps({"type": "train_episode", "step": step, "episode": episode, "return": ep_ret, "length": ep_len}) + "\n")
            metrics_f.flush()
            episode += 1
            obs_raw, _ = env.reset(seed=cfg.seed + episode)
            obs = flatten_obs(obs_raw)
            ep_ret, ep_len = 0.0, 0

        if step >= cfg.learning_starts and replay.size >= cfg.batch_size and step % cfg.update_every == 0:
            for _ in range(cfg.gradient_steps):
                last_update = update(cfg, step, replay, actor, q1, q2, q1_t, q2_t, actor_opt, q_opt, log_alpha, alpha_opt, target_entropy, device)
            if step % 1000 == 0:
                for k, v in last_update.items():
                    writer.add_scalar(f"loss/{k}", v, step)

        if step % cfg.eval_every == 0 or step == cfg.total_steps:
            ret_mean, ret_std, len_mean = evaluate(actor, cfg, device)
            fps = step / max(time.time() - start, 1e-6)
            row = {"type": "eval", "step": step, "eval_return_mean": ret_mean, "eval_return_std": ret_std, "eval_length_mean": len_mean, "fps": fps, "device": str(device)}
            row.update({f"update_{k}": v for k, v in last_update.items()})
            writer.add_scalar("eval/return_mean", ret_mean, step)
            writer.add_scalar("system/fps", fps, step)
            metrics_f.write(json.dumps(row) + "\n")
            metrics_f.flush()

    torch.save({"config": asdict(cfg), "log_path": str(run_dir), "total_steps": cfg.total_steps, "wall_time_sec": time.time() - start, "device": str(device)}, run_dir / "result.pt")
    metrics_f.close()
    writer.close()
    env.close()
    return run_dir


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--policy-lr", type=float, default=3e-4)
    p.add_argument("--q-lr", type=float, default=3e-4)
    p.add_argument("--alpha-lr", type=float, default=3e-4)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--max-episode-steps", type=int, default=1000)
    p.add_argument("--device", default="auto")
    p.add_argument("--logdir", default="logs_blcr_sac")
    p.add_argument("--per-alpha", type=float, default=0.6)
    p.add_argument("--beta-start", type=float, default=0.4)
    p.add_argument("--priority-eps", type=float, default=1e-6)
    p.add_argument("--blcr-floor-lambda", type=float, default=0.35)
    p.add_argument("--blcr-loss-eta", type=float, default=0.08)
    p.add_argument("--blcr-loss-wmax", type=float, default=1.5)
    p.add_argument("--signal-clip", type=float, default=10.0)
    p.add_argument("--no-autotune-entropy", action="store_true")
    p.add_argument("--fixed-entropy-alpha", type=float, default=0.2)
    p.add_argument("--num-threads", type=int, default=1)
    args = p.parse_args()
    data = vars(args)
    data["autotune_entropy"] = not data.pop("no_autotune_entropy")
    return TrainConfig(**data)


if __name__ == "__main__":
    train(parse_args())

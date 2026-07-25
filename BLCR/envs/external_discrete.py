from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


def _flatten_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [_flatten_obs(obs[key]) for key in sorted(obs)]
        return np.concatenate(parts, axis=0).astype(np.float32)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


class GymDiscreteVectorEnv:
    def __init__(self, env_id: str, seed: int, env_kwargs: dict[str, Any] | None = None) -> None:
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "Gymnasium is required for Gym/ALE experiments. "
                "Install the project dependencies with `pip install -r requirements.txt`."
            ) from exc

        if env_id.startswith("ALE/"):
            try:
                import ale_py

                gym.register_envs(ale_py)
            except ImportError as exc:
                raise ImportError(
                    "ale_py is required for ALE experiments. "
                    "Install the project dependencies with `pip install -r requirements.txt`."
                ) from exc

        self.env = gym.make(env_id, **(env_kwargs or {}))
        if not hasattr(self.env.action_space, "n"):
            raise ValueError(f"Environment {env_id} does not expose a discrete action space.")
        self.num_actions = int(self.env.action_space.n)
        self._seed = seed
        self._last_obs: np.ndarray | None = None

    def reset(self) -> np.ndarray:
        obs, _info = self.env.reset(seed=self._seed)
        self._seed += 1
        self._last_obs = _flatten_obs(obs)
        return self._last_obs

    def step(self, action: int) -> StepResult:
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        self._last_obs = _flatten_obs(obs)
        return StepResult(self._last_obs, float(reward), bool(terminated), bool(truncated), dict(info))

    def close(self) -> None:
        self.env.close()

    @property
    def observation_dim(self) -> int:
        if self._last_obs is None:
            self.reset()
        assert self._last_obs is not None
        return int(self._last_obs.size)


class DMControlDiscreteVectorEnv:
    def __init__(
        self,
        domain_name: str,
        task_name: str,
        seed: int,
        action_mode: str = "axis",
        action_bins: int = 5,
        max_grid_actions: int = 256,
        task_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from dm_control import suite
        except ImportError as exc:
            raise ImportError(
                "dm_control is required for DeepMind Control experiments. "
                "Install it with `pip install dm-control`."
            ) from exc

        kwargs = dict(task_kwargs or {})
        kwargs.setdefault("random", seed)
        self.env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=kwargs)
        self._action_values = self._build_action_values(action_mode, action_bins, max_grid_actions)
        self.num_actions = int(self._action_values.shape[0])
        self._last_obs: np.ndarray | None = None

    def _build_action_values(
        self, action_mode: str, action_bins: int, max_grid_actions: int
    ) -> np.ndarray:
        spec = self.env.action_spec()
        minimum = np.asarray(spec.minimum, dtype=np.float32).reshape(-1)
        maximum = np.asarray(spec.maximum, dtype=np.float32).reshape(-1)
        dim = int(minimum.size)

        if action_mode == "axis":
            zero = np.clip(np.zeros(dim, dtype=np.float32), minimum, maximum)
            actions = [zero]
            for i in range(dim):
                low = zero.copy()
                high = zero.copy()
                low[i] = minimum[i]
                high[i] = maximum[i]
                actions.extend([low, high])
            return np.stack(actions, axis=0).astype(np.float32)

        if action_mode == "grid":
            bins = [np.linspace(minimum[i], maximum[i], action_bins, dtype=np.float32) for i in range(dim)]
            total = int(np.prod([len(x) for x in bins]))
            if total > max_grid_actions:
                raise ValueError(
                    f"Grid action discretization would create {total} actions, "
                    f"which exceeds max_grid_actions={max_grid_actions}. "
                    "Use env.action_mode=axis or reduce env.action_bins."
                )
            return np.asarray(list(product(*bins)), dtype=np.float32)

        raise ValueError(f"Unknown DMControl action_mode: {action_mode}")

    def reset(self) -> np.ndarray:
        timestep = self.env.reset()
        self._last_obs = _flatten_obs(timestep.observation)
        return self._last_obs

    def step(self, action: int) -> StepResult:
        timestep = self.env.step(self._action_values[int(action)])
        obs = _flatten_obs(timestep.observation)
        self._last_obs = obs
        # dm_env uses discount=None on terminal time steps.
        terminated = timestep.last()
        truncated = False
        return StepResult(obs, float(timestep.reward or 0.0), bool(terminated), truncated, {})

    def close(self) -> None:
        self.env.close()

    @property
    def observation_dim(self) -> int:
        if self._last_obs is None:
            self.reset()
        assert self._last_obs is not None
        return int(self._last_obs.size)


def make_external_discrete_env(config: dict[str, Any], seed: int):
    env_type = str(config["env"]["type"]).lower()
    if env_type in {"gym", "ale"}:
        return GymDiscreteVectorEnv(
            env_id=str(config["env"]["id"]),
            seed=seed,
            env_kwargs=dict(config["env"].get("kwargs", {})),
        )
    if env_type in {"dmcontrol", "dmc", "dm_control"}:
        return DMControlDiscreteVectorEnv(
            domain_name=str(config["env"]["domain"]),
            task_name=str(config["env"]["task"]),
            seed=seed,
            action_mode=str(config["env"].get("action_mode", "axis")),
            action_bins=int(config["env"].get("action_bins", 5)),
            max_grid_actions=int(config["env"].get("max_grid_actions", 256)),
            task_kwargs=dict(config["env"].get("task_kwargs", {})),
        )
    raise ValueError(f"Unknown external env type: {env_type}")

from __future__ import annotations

import shutil
import sys
import tempfile
import types
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fluid.agents.dqn_trainer import DQNTrainer
from fluid.config import load_config


class FakeEnvironment:
    def __init__(self, game: str, random_seed: int = 0):
        self.game = game
        self.random = np.random.default_rng(random_seed)
        self.channels = 4
        self.max_episode_steps = 12
        self.steps = 0
        self._state = self.random.integers(0, 2, size=(10, 10, self.channels), dtype=np.uint8)

    def state_shape(self) -> tuple[int, int, int]:
        return self._state.shape

    def num_actions(self) -> int:
        return 6

    def reset(self) -> None:
        self.steps = 0
        self._state = self.random.integers(0, 2, size=(10, 10, self.channels), dtype=np.uint8)

    def state(self) -> np.ndarray:
        return self._state.copy()

    def act(self, action: int) -> tuple[float, bool]:
        del action
        self.steps += 1
        self._state = self.random.integers(0, 2, size=(10, 10, self.channels), dtype=np.uint8)
        reward = float(self.random.choice([0.0, 0.0, 0.0, 1.0]))
        terminated = self.steps >= self.max_episode_steps
        return reward, terminated


def install_fake_minatar() -> None:
    fake_module = types.ModuleType("minatar")
    fake_module.Environment = FakeEnvironment
    sys.modules["minatar"] = fake_module


def build_test_config(algo: str, log_root: Path) -> dict:
    config = load_config(Path("configs/minatar") / f"{algo}.yaml")
    config = deepcopy(config)
    config["experiment"]["game"] = "stub_world"
    config["experiment"]["seed"] = 0
    config["experiment"]["logdir"] = str(log_root)
    config["training"]["num_frames"] = 64
    config["training"]["batch_size"] = 8
    config["training"]["replay_capacity"] = 64
    config["training"]["replay_start_size"] = 8
    config["training"]["target_update_freq"] = 8
    config["training"]["train_freq"] = 1
    config["training"]["eps_decay_frames"] = 32
    config["training"]["console_log_every"] = 2
    config["logging"]["log_every"] = 16
    return config


def assert_run_outputs(log_root: Path, algo: str) -> None:
    run_dir = log_root / "stub_world" / algo / "seed0"
    assert run_dir.exists(), f"missing run directory for {algo}"
    assert (run_dir / "metrics.jsonl").exists(), f"missing metrics for {algo}"
    assert (run_dir / "result.pt").exists(), f"missing result checkpoint for {algo}"
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    assert event_files, f"missing tensorboard event file for {algo}"


def main() -> None:
    install_fake_minatar()
    temp_root = Path(tempfile.mkdtemp(prefix="blcr_smoke_"))
    try:
        for algo in ("dqn", "per", "relo", "relo_floor", "blcr"):
            trainer = DQNTrainer(build_test_config(algo, temp_root))
            trainer.train()
            assert_run_outputs(temp_root, algo)
        print(f"Smoke tests passed. Logs written under {temp_root}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import collections
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Logger:
    def __init__(self, logdir: str | Path, step: int = 0):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.logdir))
        self.scalars = collections.defaultdict(list)
        self.step = step
        self._last_step = None
        self._last_time = None

    def info(self, message: str) -> None:
        print(f"{now()} | {message}")

    def scalar(self, name: str, value: float) -> None:
        self.scalars[name].append(float(value))

    def write(self, with_fps: bool = False) -> None:
        if not self.scalars:
            return
        payload = {name: float(np.mean(values)) for name, values in self.scalars.items()}
        if with_fps:
            payload["perf/fps"] = self._compute_fps(self.step)
        with (self.logdir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": self.step, **payload}) + "\n")
        for name, value in payload.items():
            prefix = "" if "/" in name else "scalars/"
            self.writer.add_scalar(prefix + name, value, self.step)
        self.writer.flush()
        self.scalars.clear()

    def _compute_fps(self, step: int) -> float:
        if self._last_step is None:
            self._last_step = step
            self._last_time = time.time()
            return 0.0
        elapsed_steps = step - self._last_step
        elapsed_time = time.time() - self._last_time
        self._last_step = step
        self._last_time = time.time()
        return elapsed_steps / max(elapsed_time, 1e-6)

    def close(self) -> None:
        self.writer.close()

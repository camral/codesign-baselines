"""Shared process-time budgets for matched-wall-clock comparisons."""

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
import time
from typing import Callable


@dataclass
class WallClockBudget:
    seconds: float
    started_at: float
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self):
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("wall-clock budget must be finite and positive")
        if not math.isfinite(self.started_at):
            raise ValueError("start time must be finite")

    @classmethod
    def start(cls, hours: float, clock: Callable[[], float] = time.monotonic):
        started = clock()
        if clock is time.monotonic and "CODESIGN_RUN_STARTED" in os.environ:
            started = float(os.environ["CODESIGN_RUN_STARTED"])
        return cls(float(hours) * 3600, started, clock)

    @property
    def elapsed(self) -> float:
        if hasattr(self, "_modeled_elapsed_at"):
            return self._modeled_elapsed_at
        return max(0.0, self.clock() - self.started_at)

    @property
    def remaining(self) -> float:
        return max(0.0, self.seconds - self.elapsed)

    @property
    def request_remaining(self):
        return None if hasattr(self, "_modeled_elapsed_at") else self.remaining

    @property
    def expired(self) -> bool:
        return self.elapsed >= self.seconds

    @property
    def evaluation_interval(self) -> float:
        return self.seconds * 0.05

    def next_evaluation(self) -> float:
        interval = self.evaluation_interval
        return (math.floor(self.elapsed / interval) + 1) * interval

    def advance_evaluation(self, boundary: float) -> float:
        boundary += self.evaluation_interval
        while boundary <= self.elapsed:
            boundary += self.evaluation_interval
        return boundary

    @contextmanager
    def modeled_elapsed(self, seconds: float):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("modeled elapsed time must be finite and nonnegative")
        if hasattr(self, "_modeled_elapsed_at"):
            raise RuntimeError("modeled elapsed time cannot be nested")
        real_started, original_started_at = self.clock(), self.started_at
        self._modeled_elapsed_at = self.elapsed
        try:
            yield
        finally:
            real_elapsed = max(0.0, self.clock()-real_started)
            self.started_at = original_started_at+real_elapsed-seconds
            del self._modeled_elapsed_at

    def metadata(self) -> dict:
        return {
            "budget_seconds": self.seconds,
            "clock": "monotonic",
            "includes": ["initialization", "compilation", "training", "search", "evaluation"],
            "queue_time_included": False,
        }

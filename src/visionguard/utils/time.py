"""Tiny timing helpers."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


def now_s() -> float:
    """Monotonic-ish wall clock seconds."""

    return time.time()


@contextmanager
def stopwatch() -> Iterator[list[float]]:
    """Context manager that stores elapsed milliseconds in ``bucket[0]``."""

    bucket = [0.0]
    start = time.perf_counter()
    try:
        yield bucket
    finally:
        bucket[0] = (time.perf_counter() - start) * 1000.0


class Ema:
    """Exponential moving average."""

    def __init__(self, alpha: float = 0.2, initial: float = 0.0) -> None:
        self.alpha = alpha
        self.value = initial
        self._initialized = False

    def update(self, sample: float) -> float:
        """Fold a new sample into the average."""

        if not self._initialized:
            self.value = sample
            self._initialized = True
        else:
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value
        return self.value

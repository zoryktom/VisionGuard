"""Structured logging helpers."""

from __future__ import annotations

import logging
import sys
from typing import Any

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once for CLI and API processes."""

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a child logger."""

    return logging.getLogger(name)


def log_context(logger: logging.Logger, **fields: Any) -> None:
    """Emit a compact key=value debug line."""

    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.debug(payload)

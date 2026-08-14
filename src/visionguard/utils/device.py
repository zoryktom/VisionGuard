"""Device selection for GPU-ready inference and training."""

from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick CUDA, Apple MPS, or CPU.

    Args:
        preference: ``auto``, ``cuda``, ``mps``, or ``cpu``.
    """

    requested = preference.lower().strip()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Human-readable device string for dashboards and logs."""

    if device.type == "cuda":
        index = device.index or 0
        name = torch.cuda.get_device_name(index)
        return f"cuda:{index} ({name})"
    return device.type

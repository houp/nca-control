from __future__ import annotations

"""Helpers for choosing the execution device for PyTorch code paths."""

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` to the best local device, otherwise honor the explicit request."""

    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)

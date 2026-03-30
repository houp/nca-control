from __future__ import annotations

"""Helpers for choosing the execution device for PyTorch code paths."""

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` to the best local device, otherwise honor the explicit request."""

    if requested == "auto":
        # Prefer CUDA when available so Linux collaborators can use the same
        # PyTorch training/evaluation entrypoints without changing the code.
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)

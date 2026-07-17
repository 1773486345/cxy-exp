"""Parameter-free moving-average decomposition primitives."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def resolve_moving_average_window(patch_size: int, seq_len: int) -> int:
    """Return the nearest legal odd window, preferring the smaller tie."""
    if isinstance(patch_size, bool) or not isinstance(patch_size, int):
        raise TypeError("patch_size must be an integer")
    if isinstance(seq_len, bool) or not isinstance(seq_len, int):
        raise TypeError("seq_len must be an integer")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    legal_windows = range(1, seq_len + 1, 2)
    return min(legal_windows, key=lambda window: (abs(window - patch_size), window))


def moving_average(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Apply a replicate-padded moving average along the time dimension only."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, time, channel]")
    if not x.is_floating_point():
        raise TypeError("x must have a floating-point dtype")
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise TypeError("window_size must be an integer")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if window_size > x.shape[1]:
        raise ValueError("window_size must not exceed the time dimension")
    if window_size == 1:
        return x

    radius = window_size // 2
    padded = functional.pad(x.transpose(1, 2), (radius, radius), mode="replicate")
    return padded.transpose(1, 2).unfold(1, window_size, 1).mean(dim=-1)


def decompose_slow_fast(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return slow and fast components whose sum reconstructs ``x``."""
    slow = moving_average(x, window_size)
    fast = x - slow
    return slow, fast

"""Small patch and overlap-add utilities used by the v2 branches."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def patchify_time(x: Tensor, patch_size: int, stride: int) -> Tensor:
    if x.ndim != 3:
        raise ValueError("patchify_time expects [B,T,C]")
    length = x.size(1)
    count = 1 if length <= patch_size else (length - patch_size + stride - 1) // stride + 1
    padded = (count - 1) * stride + patch_size
    if padded > length:
        x = F.pad(x, (0, 0, 0, padded - length))
    return x.unfold(1, patch_size, stride).permute(0, 1, 3, 2).contiguous()


def overlap_add(patches: Tensor, seq_len: int, stride: int) -> Tensor:
    if patches.ndim != 4:
        raise ValueError("overlap_add expects [B,P,K,C]")
    batch, count, size, channels = patches.shape
    total = (count - 1) * stride + size
    output = patches.new_zeros(batch, total, channels)
    counts = patches.new_zeros(batch, total, channels)
    for index in range(count):
        start = index * stride
        output[:, start:start + size] += patches[:, index]
        counts[:, start:start + size] += 1
    return (output / counts.clamp_min(1.0))[:, :seq_len]

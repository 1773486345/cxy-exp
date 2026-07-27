#!/usr/bin/env python3
"""One-batch CUDA forward/backward smoke for the SWAT TypeFusion structure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel


def swat_config(batch_size: int) -> TypeFusionConfig:
    return TypeFusionConfig(
        seq_len=2048,
        patch_size=256,
        patch_stride=64,
        c_in=51,
        batch_size=batch_size,
        d_model=128,
        cf_dim=64,
        d_ff=256,
        n_heads=2,
        head_dim=64,
        e_layers=3,
        dropout=0.2,
        temporal_hidden_dim=128,
        temporal_layers=3,
        memory_size=32,
        memory_topk=4,
        branch_dim=128,
        fusion_layers=2,
        fusion_heads=4,
        relation_mask_groups=4,
        pattern_mask_ratio=0.25,
        training_stage="branch_pretrain",
        fit_mode="three_stage",
        training_budget_mode="equal_total_steps",
    )


def assert_finite(value: Any, name: str) -> None:
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise RuntimeError(f"non-finite tensor: {name}")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{name}.{key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available")
        return

    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    device = torch.device("cuda")
    config = swat_config(args.batch_size)
    model = TypeFusionCATCHModel(config).to(device).train()
    x = torch.randn(args.batch_size, config.seq_len, config.c_in, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    output = model(x, compute_joint=False)
    assert_finite(output["branches"], "branches")
    assert_finite(output["losses"], "losses")
    output["losses"]["total"].backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"missing_or_nonfinite_gradient:{name}")
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(f"SWAT CUDA smoke passed: batch_size={args.batch_size}, peak_memory_mib={peak_mib:.2f}")


if __name__ == "__main__":
    main()

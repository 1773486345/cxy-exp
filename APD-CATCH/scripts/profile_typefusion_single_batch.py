#!/usr/bin/env python3
"""Profile one random TypeFusion-CATCH training batch without benchmark I/O."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel


def psm_config(batch_size: int) -> TypeFusionConfig:
    """PSM's registered task structure, with no dataset access."""

    return TypeFusionConfig(
        seq_len=192,
        patch_size=16,
        patch_stride=8,
        c_in=25,
        batch_size=batch_size,
        d_model=16,
        cf_dim=16,
        d_ff=32,
        n_heads=4,
        head_dim=32,
        e_layers=1,
        dropout=0.3,
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


class ModuleTimer:
    """Forward hooks for actual branch calls; CUDA sync is profiling-only."""

    def __init__(self, device: torch.device, synchronize: bool) -> None:
        self.device = device
        self.synchronize = synchronize
        self._starts: Dict[str, List[float]] = {}
        self.elapsed: Dict[str, float] = {}
        self._handles = []

    def _sync(self) -> None:
        if self.synchronize:
            torch.cuda.synchronize(self.device)

    def add(self, name: str, module: torch.nn.Module) -> None:
        def before(_module, _inputs) -> None:
            self._sync()
            self._starts.setdefault(name, []).append(time.perf_counter())

        def after(_module, _inputs, _output) -> None:
            self._sync()
            started = self._starts[name].pop()
            self.elapsed[name] = self.elapsed.get(name, 0.0) + time.perf_counter() - started

        self._handles.append(module.register_forward_pre_hook(before))
        self._handles.append(module.register_forward_hook(after))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("PSM",), default="PSM")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available", flush=True)
        return

    # This tool profiles PSM only; keeping the task choice explicit makes the
    # registered dimensions visible at invocation time.
    config = psm_config(args.batch_size)
    device = torch.device("cuda")
    synchronize = os.environ.get("TYPEFUSION_PROFILE_TIMING") == "1"
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    model = TypeFusionCATCHModel(config).to(device).train()
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=config.lr
    )
    x = torch.randn(args.batch_size, config.seq_len, config.c_in, device=device)
    relation = model.relation_branch
    condition_batch_chunk = relation._condition_batch_chunk(config.seq_len)
    condition_calls = relation.max_groups * math.ceil(args.batch_size / condition_batch_chunk)
    print(
        "relation_profile: "
        f"task={args.task} group_count={relation.max_groups} "
        f"condition_batch_chunk={condition_batch_chunk} "
        f"condition_forward_calls={condition_calls} "
        f"max_attention_rows={condition_batch_chunk * config.seq_len} "
        f"profile_timing={synchronize}",
        flush=True,
    )

    timer = ModuleTimer(device, synchronize=synchronize)
    for name in ("shared_stem", "state_branch", "evolution_branch", "pattern_branch", "relation_branch"):
        timer.add(name, getattr(model, name))
    torch.cuda.reset_peak_memory_stats(device)
    if synchronize:
        torch.cuda.synchronize(device)
    total_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    output = model(x, compute_joint=False)
    if synchronize:
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - total_started
    if not torch.isfinite(output["losses"]["total"]):
        raise RuntimeError("non-finite total loss")

    if synchronize:
        torch.cuda.synchronize(device)
    backward_started = time.perf_counter()
    output["losses"]["total"].backward()
    if synchronize:
        torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_started

    if synchronize:
        torch.cuda.synchronize(device)
    optimizer_started = time.perf_counter()
    optimizer.step()
    if synchronize:
        torch.cuda.synchronize(device)
    optimizer_seconds = time.perf_counter() - optimizer_started
    timer.close()

    for name in ("shared_stem", "state_branch", "evolution_branch", "pattern_branch", "relation_branch"):
        print(f"{name}_seconds={timer.elapsed.get(name, 0.0):.6f}", flush=True)
    print(f"forward_seconds={forward_seconds:.6f}", flush=True)
    print(f"backward_seconds={backward_seconds:.6f}", flush=True)
    print(f"optimizer_seconds={optimizer_seconds:.6f}", flush=True)
    print(
        "cuda_memory_mib: "
        f"allocated={torch.cuda.memory_allocated(device) / (1024 * 1024):.2f} "
        f"reserved={torch.cuda.memory_reserved(device) / (1024 * 1024):.2f} "
        f"peak={torch.cuda.max_memory_allocated(device) / (1024 * 1024):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

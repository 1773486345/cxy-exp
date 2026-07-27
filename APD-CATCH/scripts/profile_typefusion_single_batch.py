#!/usr/bin/env python3
"""Profile one random TypeFusion-CATCH training batch without benchmark I/O."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel


_TASK_CONFIGS = {
    # These immutable structures are copied from the prepared single-task
    # scripts and data registry. The profiler never reads task data or writes
    # a benchmark artifact.
    "PSM": {
        "batch_size": 128,
        "seq_len": 192,
        "patch_size": 16,
        "patch_stride": 8,
        "c_in": 25,
        "d_model": 16,
        "cf_dim": 16,
        "d_ff": 32,
        "n_heads": 4,
        "head_dim": 32,
        "e_layers": 1,
        "dropout": 0.3,
    },
    "CICIDS": {
        "batch_size": 128,
        "seq_len": 192,
        "patch_size": 16,
        "patch_stride": 16,
        "c_in": 72,
        "d_model": 128,
        "cf_dim": 64,
        "d_ff": 128,
        "n_heads": 16,
        "head_dim": 16,
        "e_layers": 3,
        "dropout": 0.2,
    },
    "MSL": {
        "batch_size": 128,
        "seq_len": 192,
        "patch_size": 16,
        "patch_stride": 8,
        "c_in": 55,
        "d_model": 128,
        "cf_dim": 64,
        "d_ff": 256,
        "n_heads": 2,
        "head_dim": 64,
        "e_layers": 3,
        "dropout": 0.2,
    },
    "ASD_dataset_1": {
        "batch_size": 128,
        "seq_len": 192,
        "patch_size": 16,
        "patch_stride": 16,
        "c_in": 26,
        "d_model": 256,
        "cf_dim": 4,
        "d_ff": 256,
        "n_heads": 4,
        "head_dim": 32,
        "e_layers": 3,
        "dropout": 0.2,
    },
}


def assert_finite(value, name: str) -> None:
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise RuntimeError(f"non-finite tensor: {name}")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{name}.{key}")


def task_config(task: str, batch_size: int | None) -> TypeFusionConfig:
    """Construct a registered task structure without dataset access."""

    task_params = dict(_TASK_CONFIGS[task])
    if batch_size is not None:
        task_params["batch_size"] = batch_size
    return TypeFusionConfig(
        **task_params,
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
    parser.add_argument("--task", choices=tuple(_TASK_CONFIGS), default="PSM")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="defaults to the selected task's registered formal batch size",
    )
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available", flush=True)
        return

    config = task_config(args.task, args.batch_size)
    device = torch.device("cuda")
    synchronize = os.environ.get("TYPEFUSION_PROFILE_TIMING") == "1"
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    model = TypeFusionCATCHModel(config).to(device).train()
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=config.lr
    )
    x = torch.randn(config.batch_size, config.seq_len, config.c_in, device=device)
    relation = model.relation_branch
    condition_batch_chunk = relation._condition_batch_chunk(config.seq_len)
    condition_calls = relation.max_groups * math.ceil(config.batch_size / condition_batch_chunk)
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
    assert_finite(output["branches"], "branches")
    assert_finite(output["losses"], "losses")

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

    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"missing_or_nonfinite_gradient:{name}")

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

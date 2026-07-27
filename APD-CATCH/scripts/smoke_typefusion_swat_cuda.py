#!/usr/bin/env python3
"""One-batch CUDA localization smoke for the SWAT TypeFusion structure."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.layers.losses import compute_losses
from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel


_STAGES = ("branch_pretrain", "fusion_train", "joint_finetune")


def swat_config(batch_size: int, stage: str) -> TypeFusionConfig:
    """Return the fixed SWAT task structure without reading the dataset."""

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
        training_stage=stage,
        fit_mode="three_stage",
        training_budget_mode="equal_total_steps",
    )


def assert_finite(value: Any, name: str) -> None:
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise RuntimeError(f"non-finite tensor: {name}")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{name}.{key}")


class CudaTracer:
    """Emit synchronized boundaries only for this explicit debug tool."""

    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.device = device
        self.enabled = enabled

    def mark(self, name: str) -> None:
        if self.enabled:
            torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
        peak = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        print(
            f"{name}: timestamp={time.perf_counter():.6f} "
            f"allocated_mib={allocated:.2f} reserved_mib={reserved:.2f} peak_mib={peak:.2f}",
            flush=True,
        )

    def run(self, name: str, function: Callable[[], Any]) -> Any:
        self.mark(f"{name}_start")
        try:
            result = function()
        except Exception:
            self.mark(f"{name}_error")
            raise
        self.mark(f"{name}_end")
        return result


def _profiled_forward(
    model: TypeFusionCATCHModel,
    x: torch.Tensor,
    tracer: CudaTracer,
) -> Dict[str, object]:
    """Execute the production modules one at a time for fault localization."""

    shared = tracer.run("shared_stem", lambda: model.shared_stem(x))
    normalized_input = shared["normalized_input"]
    randomize_groups = model.training and model.config.training_stage != "fusion_train"
    branches = {
        "state": tracer.run(
            "state_branch",
            lambda: model.state_branch(normalized_input, shared["temporal_latent"]),
        ),
        "evolution": tracer.run("evolution_branch", lambda: model.evolution_branch(x)),
        "pattern": tracer.run(
            "pattern_branch",
            lambda: model.pattern_branch(
                normalized_input,
                shared["frequency_channels"],
                training_mask=model.training and model.config.training_stage != "fusion_train",
            ),
        ),
        "relation": tracer.run(
            "relation_branch",
            lambda: model.relation_branch(
                normalized_input,
                randomize_groups=randomize_groups,
                debug_callback=tracer.mark,
            ),
        ),
    }

    if model.config.training_stage == "branch_pretrain":
        output: Dict[str, object] = {
            "normalized_input": normalized_input,
            "evolution_input": x,
            "spectrum": shared["spectrum"],
            "branches": branches,
            "q": None,
            "q_normal": None,
            "branch_mask_prediction": None,
            "branch_mask_loss": normalized_input.new_zeros(()),
            "leave_one_out": None,
            "x_hat_joint_normalized": None,
            "x_hat_joint": None,
            "total_score": None,
            "branch_conflict_map": None,
        }
    else:
        q = tracer.run("evidence_adapter", lambda: model._adapt_evidence(branches))
        leave_one_out = tracer.run("leave_one_out", lambda: model.branch_fusion.leave_one_out(q))
        mask_prediction = tracer.run(
            "branch_mask_prediction",
            lambda: model.branch_fusion.masked_branch_prediction(q),
        )
        q_normal = leave_one_out["q_normal"]
        x_hat_joint_normalized = tracer.run(
            "joint_decoder", lambda: model.joint_decoder(q_normal)
        )
        x_hat_joint = shared["revin"].denormalize(
            x_hat_joint_normalized, shared["revin_statistics"]
        )
        output = {
            "normalized_input": normalized_input,
            "evolution_input": x,
            "spectrum": shared["spectrum"],
            "branches": branches,
            "q": q,
            "q_normal": q_normal,
            "branch_mask_prediction": mask_prediction,
            "branch_mask_loss": mask_prediction["loss"],
            "leave_one_out": leave_one_out,
            "x_hat_joint_normalized": x_hat_joint_normalized,
            "x_hat_joint": x_hat_joint,
            "total_score": (x - x_hat_joint).abs(),
            "branch_conflict_map": (q - q_normal).abs().mean(dim=-1),
        }

    output["losses"] = tracer.run("loss", lambda: compute_losses(output, model.config))
    tracer.mark("relation_loss_start")
    _ = output["losses"]["relation"]
    tracer.mark("relation_loss_end")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stage", choices=_STAGES, default="branch_pretrain")
    parser.add_argument(
        "--profile-branches",
        action="store_true",
        help="execute named production modules separately and synchronize at each boundary",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available", flush=True)
        return

    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    device = torch.device("cuda")
    config = swat_config(args.batch_size, args.stage)
    model = TypeFusionCATCHModel(config).to(device).train()
    x = torch.randn(args.batch_size, config.seq_len, config.c_in, device=device)
    relation = model.relation_branch
    condition_batch_chunk = relation._condition_batch_chunk(config.seq_len)
    condition_calls = relation.max_groups * math.ceil(args.batch_size / condition_batch_chunk)
    print(
        "relation_conditions: "
        f"group_count={relation.max_groups} condition_batch_chunk={condition_batch_chunk} "
        f"condition_forward_calls={condition_calls} max_attention_rows={condition_batch_chunk * config.seq_len}",
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats(device)
    profile_branches = args.profile_branches or os.environ.get("TYPEFUSION_PROFILE_TIMING") == "1"
    tracer = CudaTracer(device, enabled=profile_branches)
    if args.profile_branches:
        output = _profiled_forward(model, x, tracer)
    else:
        output = tracer.run(
            "model_forward", lambda: model(x, compute_joint=args.stage != "branch_pretrain")
        )
    assert_finite(output["branches"], "branches")
    assert_finite(output["losses"], "losses")
    tracer.mark("backward_start")
    tracer.mark("relation_backward_start")
    output["losses"]["total"].backward()
    tracer.mark("relation_backward_end")
    tracer.mark("backward_end")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"missing_or_nonfinite_gradient:{name}")
    print(
        "SWAT CUDA smoke passed: "
        f"batch_size={args.batch_size} stage={args.stage} "
        f"peak_memory_mib={torch.cuda.max_memory_allocated(device) / (1024 * 1024):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

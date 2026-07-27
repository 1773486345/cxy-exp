"""One-batch random-tensor implementation smoke test; it reports no performance."""

from __future__ import annotations

import torch

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel


def main() -> None:
    torch.manual_seed(0)
    config = TypeFusionConfig(
        seq_len=32,
        patch_size=8,
        patch_stride=4,
        c_in=4,
        d_model=32,
        cf_dim=32,
        d_ff=64,
        n_heads=2,
        e_layers=1,
        temporal_hidden_dim=32,
        temporal_layers=2,
        memory_size=8,
        memory_topk=2,
        branch_dim=32,
        fusion_layers=1,
        fusion_heads=4,
        relation_mask_groups=2,
        dropout=0.0,
        training_stage="joint_finetune",
    )
    model = TypeFusionCATCHModel(config).train()
    x = torch.randn(2, config.seq_len, config.c_in)
    output = model(x)
    output["losses"]["total"].backward()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters={parameter_count}")
    print(f"x_hat_joint_shape={tuple(output['x_hat_joint'].shape)}")
    print(f"total_score_finite={bool(torch.isfinite(output['total_score']).all())}")
    print(f"loss_finite={bool(torch.isfinite(output['losses']['total']))}")


if __name__ == "__main__":
    main()

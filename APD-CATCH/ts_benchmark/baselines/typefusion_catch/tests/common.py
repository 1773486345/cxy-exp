"""Small CPU configurations shared by TypeFusion-CATCH unit tests."""

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig


def tiny_config(training_stage: str = "joint_finetune") -> TypeFusionConfig:
    return TypeFusionConfig(
        seq_len=32,
        patch_size=8,
        patch_stride=4,
        c_in=4,
        d_model=32,
        cf_dim=32,
        d_ff=64,
        n_heads=2,
        head_dim=16,
        e_layers=1,
        dropout=0.0,
        temporal_hidden_dim=32,
        temporal_layers=2,
        memory_size=8,
        memory_topk=2,
        branch_dim=32,
        fusion_layers=1,
        fusion_heads=4,
        relation_mask_groups=2,
        pattern_mask_ratio=0.25,
        training_stage=training_stage,
    )

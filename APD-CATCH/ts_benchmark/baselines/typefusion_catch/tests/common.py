"""Small CPU configurations and normal DataFrames for unit tests."""

import numpy as np
import pandas as pd

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


def tiny_fit_kwargs(**overrides):
    kwargs = {
        "seq_len": 8,
        "patch_size": 4,
        "patch_stride": 2,
        "d_model": 16,
        "cf_dim": 16,
        "d_ff": 32,
        "n_heads": 2,
        "head_dim": 8,
        "e_layers": 1,
        "dropout": 0.0,
        "temporal_hidden_dim": 16,
        "temporal_layers": 1,
        "memory_size": 4,
        "memory_topk": 2,
        "branch_dim": 16,
        "fusion_layers": 1,
        "fusion_heads": 4,
        "relation_mask_groups": 2,
        "pattern_mask_ratio": 0.25,
        "batch_size": 64,
        "patience": 1,
        "branch_pretrain_epochs": 1,
        "fusion_train_epochs": 1,
        "joint_finetune_epochs": 1,
        "seed": 2021,
    }
    kwargs.update(overrides)
    return kwargs


def small_normal_frame(rows: int = 64, columns: int = 3) -> pd.DataFrame:
    time = np.arange(rows, dtype=np.float32)
    values = np.stack(
        [np.sin(time / (index + 2.0)) + 0.01 * index * time for index in range(columns)], axis=1
    )
    return pd.DataFrame(values, columns=[f"c{index}" for index in range(columns)])

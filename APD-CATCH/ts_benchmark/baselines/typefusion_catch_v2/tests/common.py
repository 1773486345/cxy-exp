from __future__ import annotations

import torch

from ts_benchmark.baselines.typefusion_catch_v2.config import TypeFusionCATCHV2Config
from ts_benchmark.baselines.typefusion_catch_v2.typefusion_catch_v2 import TypeFusionCATCHV2Model


def tiny_config(**updates) -> TypeFusionCATCHV2Config:
    values = dict(
        seed=2021, seq_len=16, patch_size=4, patch_stride=2, batch_size=2,
        c_in=4, d_model=8, cf_dim=4, d_ff=8, e_layers=1, n_heads=2,
        branch_dim=8, temporal_layers=2, joint_dim=8, joint_layers=1,
        joint_heads=2, state_memory_size=8, state_topk=2,
        relation_mask_groups=2, max_relation_attention_rows=32,
        dropout=0.0, num_epochs=1, patience=1, use_activation_checkpoint=False,
    )
    values.update(updates)
    return TypeFusionCATCHV2Config(**values)


def make_model(**updates) -> TypeFusionCATCHV2Model:
    return TypeFusionCATCHV2Model(tiny_config(**updates))

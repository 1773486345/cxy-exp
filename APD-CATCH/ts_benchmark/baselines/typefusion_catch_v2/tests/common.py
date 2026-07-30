from __future__ import annotations

import torch
from torch import nn

from ts_benchmark.baselines.typefusion_catch_v2.config import TypeFusionCATCHV2Config
from ts_benchmark.baselines.typefusion_catch_v2.typefusion_catch_v2 import TypeFusionCATCHV2Model


class FakeCatch(nn.Module):
    def __init__(self, c_in: int) -> None:
        super().__init__()
        self.projection = nn.Linear(c_in, c_in)

    def forward(self, x: torch.Tensor):
        reconstruction = self.projection(x)
        spectrum = torch.complex(reconstruction, torch.zeros_like(reconstruction))
        return reconstruction, spectrum, reconstruction.new_zeros(())


def tiny_config(**kwargs) -> TypeFusionCATCHV2Config:
    values = dict(
        seq_len=16, patch_size=4, patch_stride=2, c_in=4, batch_size=2,
        d_model=8, cf_dim=4, d_ff=8, e_layers=1, n_heads=2, head_dim=2,
        branch_dim=8, joint_dim=8, joint_layers=1, joint_heads=2,
        state_memory_size=8, state_topk=2, relation_mask_groups=2,
        temporal_layers=2, type_train_epochs=1, catch_train_epochs=1,
        dropout=0.0, use_activation_checkpoint=False,
    )
    values.update(kwargs)
    return TypeFusionCATCHV2Config(**values)


def make_model(**kwargs) -> TypeFusionCATCHV2Model:
    config = tiny_config(**kwargs)
    return TypeFusionCATCHV2Model(config, anchor_model=FakeCatch(config.c_in))

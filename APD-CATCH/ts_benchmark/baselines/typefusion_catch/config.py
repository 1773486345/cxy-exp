"""Configuration for TypeFusion-CATCH v1.

The CATCH defaults are retained where they describe the common benchmark
protocol.  New fields are explicit rather than data-set dependent.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


_STAGES = {"branch_pretrain", "fusion_train", "joint_finetune"}


@dataclass
class TypeFusionConfig:
    # CATCH-compatible defaults.
    lr: float = 1e-4
    e_layers: int = 3
    n_heads: int = 2
    cf_dim: int = 64
    d_ff: int = 256
    d_model: int = 128
    head_dim: int = 64
    dropout: float = 0.2
    batch_size: int = 128
    num_epochs: int = 3
    patience: int = 3
    seq_len: int = 192
    patch_size: int = 16
    patch_stride: int = 8
    revin: int = 1
    affine: int = 0
    subtract_last: int = 0
    seed: int = 42

    # Set by the benchmark adapter after inspecting the training frame.
    c_in: int = 1

    # TypeFusion-CATCH v1 parameters.
    temporal_hidden_dim: int = 128
    temporal_layers: int = 3
    memory_size: int = 32
    memory_topk: int = 4
    branch_dim: int = 128
    fusion_layers: int = 2
    fusion_heads: int = 4
    relation_mask_groups: int = 4
    pattern_mask_ratio: float = 0.25
    training_stage: str = "branch_pretrain"
    lambda_freq: float = 0.1
    lambda_mask: float = 0.1

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "TypeFusionConfig":
        # ``stride`` is accepted as a clear public alias while retaining the
        # CATCH spelling used by existing benchmark configurations.
        if "stride" in kwargs:
            if "patch_stride" in kwargs:
                raise ValueError("Specify either stride or patch_stride, not both")
            kwargs["patch_stride"] = kwargs.pop("stride")
        known = {field.name for field in fields(cls)}
        unknown = set(kwargs) - known
        if unknown:
            raise TypeError("Unknown TypeFusion-CATCH parameters: " + ", ".join(sorted(unknown)))
        return cls(**kwargs)

    @property
    def stride(self) -> int:
        return self.patch_stride

    @property
    def num_patches(self) -> int:
        remaining = self.seq_len - self.patch_size
        return 1 if remaining <= 0 else (remaining + self.patch_stride - 1) // self.patch_stride + 1

    def validate(self) -> None:
        integer_fields = {
            "seq_len": self.seq_len,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "c_in": self.c_in,
            "d_model": self.d_model,
            "cf_dim": self.cf_dim,
            "n_heads": self.n_heads,
            "fusion_heads": self.fusion_heads,
            "e_layers": self.e_layers,
            "fusion_layers": self.fusion_layers,
            "temporal_hidden_dim": self.temporal_hidden_dim,
            "temporal_layers": self.temporal_layers,
            "memory_size": self.memory_size,
            "memory_topk": self.memory_topk,
            "branch_dim": self.branch_dim,
            "relation_mask_groups": self.relation_mask_groups,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or value <= 0:
                raise TypeError(f"{name} must be a positive int, got {value!r}")
        float_fields = {
            "lr": self.lr,
            "dropout": self.dropout,
            "pattern_mask_ratio": self.pattern_mask_ratio,
            "lambda_freq": self.lambda_freq,
            "lambda_mask": self.lambda_mask,
        }
        for name, value in float_fields.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number, got {value!r}")
        if self.seq_len < self.patch_size:
            raise ValueError("seq_len must be at least patch_size")
        if self.cf_dim % self.n_heads != 0:
            raise ValueError("cf_dim must be divisible by n_heads")
        if self.d_model % self.fusion_heads != 0:
            raise ValueError("d_model must be divisible by fusion_heads")
        if self.branch_dim % self.fusion_heads != 0:
            raise ValueError("branch_dim must be divisible by fusion_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= float(self.pattern_mask_ratio) < 1.0:
            raise ValueError("pattern_mask_ratio must be in [0, 1)")
        if self.memory_topk > self.memory_size:
            raise ValueError("memory_topk cannot exceed memory_size")
        if self.training_stage not in _STAGES:
            raise ValueError(f"training_stage must be one of {sorted(_STAGES)}")


DEFAULT_TYPEFUSION_HYPER_PARAMS: Dict[str, Any] = {
    field.name: field.default for field in fields(TypeFusionConfig)
}

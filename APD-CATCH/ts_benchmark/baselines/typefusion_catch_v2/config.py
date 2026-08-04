"""Actual parameters for the single-stage TypeFusion-CATCH v2 model."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class TypeFusionCATCHV2Config:
    seed: int = 2021
    seq_len: int = 192
    patch_size: int = 16
    patch_stride: int = 8
    batch_size: int = 128
    lr: float = 1e-4
    num_epochs: int = 3
    patience: int = 3
    c_in: int = 1
    d_model: int = 128
    cf_dim: int = 64
    d_ff: int = 256
    e_layers: int = 3
    n_heads: int = 2
    dropout: float = 0.2

    state_memory_size: int = 32
    state_topk: int = 4
    prototype_temperature: float = 1.0
    branch_dim: int = 128
    temporal_layers: int = 3

    joint_dim: int = 128
    joint_layers: int = 2
    joint_heads: int = 4
    relation_temporal_kernel: int = 5
    relation_mask_groups: int = 4
    max_relation_attention_rows: int = 2048
    use_activation_checkpoint: bool = True

    sufficient_temperature: float = 1.0
    relation_correction_cap: float = 2.0
    responsibility_margin: float = 0.2
    score_margin: float = 0.5
    synergy_margin: float = 0.2

    lambda_task: float = 1.0
    lambda_evidence: float = 0.5
    lambda_responsibility: float = 0.5
    lambda_score: float = 1.0
    lambda_score_rank: float = 0.25
    lambda_clean_score: float = 0.1
    lambda_synergy: float = 0.25
    lambda_state_usage: float = 0.01
    lambda_pattern_frequency: float = 0.1
    loss_version: str = "typefusion_catch_v2_task_decomposition_joint_score_v2"

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "TypeFusionCATCHV2Config":
        if "stride" in kwargs:
            if "patch_stride" in kwargs:
                raise ValueError("Specify either stride or patch_stride, not both")
            kwargs["patch_stride"] = kwargs.pop("stride")
        known = {field.name for field in fields(cls)}
        unknown = set(kwargs) - known
        if unknown:
            raise TypeError("Unknown TypeFusion-CATCH v2 parameters: " + ", ".join(sorted(unknown)))
        return cls(**kwargs)

    @property
    def stride(self) -> int:
        return self.patch_stride

    @property
    def num_patches(self) -> int:
        if self.seq_len <= self.patch_size:
            return 1
        return (self.seq_len - self.patch_size + self.patch_stride - 1) // self.patch_stride + 1

    def validate(self) -> None:
        integer_fields = (
            "seq_len", "patch_size", "patch_stride", "batch_size", "num_epochs", "patience",
            "c_in", "d_model", "cf_dim", "d_ff", "e_layers", "n_heads", "state_memory_size",
            "state_topk", "branch_dim", "temporal_layers", "joint_dim", "joint_layers",
            "joint_heads", "relation_temporal_kernel", "relation_mask_groups",
            "max_relation_attention_rows",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise TypeError(f"{name} must be a positive int, got {value!r}")
        if self.seq_len < self.patch_size:
            raise ValueError("seq_len must be at least patch_size")
        if self.cf_dim % self.n_heads:
            raise ValueError("cf_dim must be divisible by n_heads")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.branch_dim % 4:
            raise ValueError("branch_dim must be divisible by the fixed four-head branch encoder")
        if self.joint_dim % self.joint_heads:
            raise ValueError("joint_dim must be divisible by joint_heads")
        if not 0 <= float(self.dropout) < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.state_topk > self.state_memory_size:
            raise ValueError("state_topk cannot exceed state_memory_size")
        if self.relation_temporal_kernel % 2 == 0:
            raise ValueError("relation_temporal_kernel must be odd")
        if self.prototype_temperature <= 0 or self.sufficient_temperature <= 0:
            raise ValueError("temperatures must be positive")
        for name in (
            "lr", "relation_correction_cap", "responsibility_margin", "score_margin", "synergy_margin",
            "lambda_task", "lambda_evidence", "lambda_responsibility", "lambda_score",
            "lambda_score_rank", "lambda_clean_score", "lambda_synergy", "lambda_state_usage",
            "lambda_pattern_frequency",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")


DEFAULT_TYPEFUSION_CATCH_V2_HYPER_PARAMS: Dict[str, Any] = {
    field.name: field.default for field in fields(TypeFusionCATCHV2Config)
}

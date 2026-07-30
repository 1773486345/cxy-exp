"""Configuration for the independent TypeFusion-CATCH v2 model."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict


@dataclass
class TypeFusionCATCHV2Config:
    # Public CATCH protocol fields.
    seed: int = 2021
    seq_len: int = 192
    patch_size: int = 16
    patch_stride: int = 8
    batch_size: int = 128
    lr: float = 1e-4
    num_epochs: int = 3
    catch_train_epochs: int = 3
    type_train_epochs: int | None = None
    patience: int = 3
    d_model: int = 128
    cf_dim: int = 64
    d_ff: int = 256
    e_layers: int = 3
    n_heads: int = 2
    head_dim: int = 64
    dropout: float = 0.2
    c_in: int = 1
    # CATCH Phase A fields. They are passed through without changing CATCH.
    Mlr: float = 1e-5
    auxi_lambda: float = 0.005
    dc_lambda: float = 0.005
    score_lambda: float = 0.05
    inference_patch_size: int = 32
    inference_patch_stride: int = 1
    anomaly_ratio: Any = field(default_factory=lambda: [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25])
    temperature: float = 0.07
    head_dropout: float = 0.1
    itr: int = 1
    small_kernel_merged: bool = True
    use_multi_scale: bool = False
    regular_lambda: float = 0.5
    pct_start: float = 0.3
    affine: int = 0
    subtract_last: int = 0
    revin: int = 1
    # v2 fixed architecture.
    state_memory_size: int = 32
    state_topk: int = 4
    branch_dim: int = 128
    temporal_layers: int = 3
    joint_dim: int = 128
    joint_layers: int = 2
    joint_heads: int = 4
    relation_mask_groups: int = 4
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
    loss_version: str = "typefusion_catch_v2_joint_score_v1"
    # Test/debug controls. No data-set-specific defaults are used.
    use_activation_checkpoint: bool = True
    anchor_state_dict: Any = None
    anchor_adapter: Any = None
    skip_anchor_fit: bool = False

    def __post_init__(self) -> None:
        # Phase B has one explicit budget rule: its epoch count follows the
        # task's CATCH Phase A epoch count unless a caller deliberately records
        # an explicit debug value.
        if self.type_train_epochs is None:
            self.type_train_epochs = self.catch_train_epochs

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
        positive = (
            "seq_len", "patch_size", "patch_stride", "batch_size", "catch_train_epochs",
            "type_train_epochs", "patience", "d_model", "cf_dim", "d_ff", "e_layers",
            "n_heads", "head_dim", "c_in", "state_memory_size", "state_topk", "branch_dim",
            "joint_dim", "joint_layers", "joint_heads", "relation_mask_groups", "temporal_layers",
        )
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise TypeError(f"{name} must be a positive int, got {value!r}")
        if self.seq_len < self.patch_size:
            raise ValueError("seq_len must be at least patch_size")
        if self.cf_dim % self.n_heads:
            raise ValueError("cf_dim must be divisible by n_heads")
        if self.joint_dim % self.joint_heads:
            raise ValueError("joint_dim must be divisible by joint_heads")
        if self.branch_dim % 4:
            raise ValueError("branch_dim must be divisible by the fixed four-head branch encoders")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.state_topk > self.state_memory_size:
            raise ValueError("state_topk cannot exceed state_memory_size")
        for name in (
            "lr", "Mlr", "auxi_lambda", "dc_lambda", "score_lambda", "lambda_task",
            "lambda_evidence", "lambda_responsibility", "lambda_score", "lambda_score_rank",
            "lambda_clean_score", "lambda_synergy", "lambda_state_usage", "lambda_pattern_frequency",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")


DEFAULT_TYPEFUSION_CATCH_V2_HYPER_PARAMS: Dict[str, Any] = {
    field.name: field.default for field in fields(TypeFusionCATCHV2Config)
}

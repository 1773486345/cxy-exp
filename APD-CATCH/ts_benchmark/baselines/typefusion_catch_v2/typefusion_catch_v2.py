"""PyTorch model for the independent TypeFusion-CATCH v2 protocol."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn

from ts_benchmark.baselines.catch.CATCH import TransformerConfig
from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel

from .config import TypeFusionCATCHV2Config
from .layers import (
    AnchorContextEncoder,
    CATCHAnchor,
    EvidenceAdapter,
    EvolutionNormalityBranch,
    PatternNormalityBranch,
    RelationAwareJointScorer,
    RelationNormalityBranch,
    StateNormalityBranch,
    TypeInterventionGenerator,
)
from .losses import compute_phase_b_losses


def make_catch_config(config: TypeFusionCATCHV2Config) -> TransformerConfig:
    """Create the original CATCH config without modifying original files."""

    catch = TransformerConfig(
        lr=config.lr, Mlr=config.Mlr, e_layers=config.e_layers, n_heads=config.n_heads,
        cf_dim=config.cf_dim, d_ff=config.d_ff, d_model=config.d_model,
        head_dim=config.head_dim, dropout=config.dropout, head_dropout=config.head_dropout,
        auxi_lambda=config.auxi_lambda, score_lambda=config.score_lambda,
        regular_lambda=config.regular_lambda, temperature=config.temperature,
        patch_stride=config.patch_stride, patch_size=config.patch_size,
        inference_patch_stride=config.inference_patch_stride,
        inference_patch_size=config.inference_patch_size, dc_lambda=config.dc_lambda,
        num_epochs=config.catch_train_epochs, batch_size=config.batch_size,
        patience=config.patience, seq_len=config.seq_len, affine=config.affine,
        subtract_last=config.subtract_last, revin=config.revin,
    )
    catch.c_in = config.c_in
    catch.enc_in = config.c_in
    catch.dec_in = config.c_in
    catch.c_out = config.c_in
    catch.label_len = 48
    catch.task_name = "anomaly_detection"
    catch.individual = 0
    return catch


class TypeFusionCATCHV2Model(nn.Module):
    branch_names = ("state", "evolution", "pattern", "relation")

    def __init__(self, config: TypeFusionCATCHV2Config, anchor_model: Optional[nn.Module] = None) -> None:
        super().__init__()
        config.validate()
        self.config = config
        if anchor_model is None:
            anchor_model = CATCHModel(make_catch_config(config))
        self.anchor = CATCHAnchor(catch_model=anchor_model)
        self.anchor_context_encoder = AnchorContextEncoder(config.c_in, config.joint_dim, config.branch_dim, config.dropout)
        self.state_branch = StateNormalityBranch(config.c_in, config.branch_dim, config.state_memory_size, config.state_topk, config.dropout, config.lambda_state_usage)
        self.evolution_branch = EvolutionNormalityBranch(config.c_in, config.branch_dim, config.temporal_layers, config.dropout)
        self.pattern_branch = PatternNormalityBranch(config.c_in, config.patch_size, config.patch_stride, config.branch_dim, config.dropout, config.lambda_pattern_frequency)
        self.relation_branch = RelationNormalityBranch(config.c_in, config.relation_mask_groups, config.branch_dim, config.dropout, config.use_activation_checkpoint)
        self.evidence_adapters = nn.ModuleList(
            EvidenceAdapter(config.branch_dim, config.joint_dim, config.joint_dim, i, config.dropout)
            for i in range(4)
        )
        self.joint_scorer = RelationAwareJointScorer(
            config.joint_dim, config.joint_layers, config.joint_heads, config.joint_dim,
            config.sufficient_temperature, config.relation_correction_cap,
        )
        self.intervention_generator = TypeInterventionGenerator(config.seed)
        self.freeze_anchor()

    def freeze_anchor(self) -> None:
        self.anchor.freeze()

    def load_anchor_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True) -> None:
        self.anchor.model.load_state_dict(state_dict, strict=strict)
        self.freeze_anchor()

    def set_phase_b_trainable(self) -> None:
        self.freeze_anchor()
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(not name.startswith("anchor."))

    def train(self, mode: bool = True) -> "TypeFusionCATCHV2Model":
        super().train(mode)
        self.anchor.eval()
        return self

    def _type_forward(self, x: Tensor, context: Tensor) -> Dict[str, Dict[str, Tensor]]:
        branches = {
            "state": self.state_branch(x, context),
            "evolution": self.evolution_branch(x),
            "pattern": self.pattern_branch(x, context),
            "relation": self.relation_branch(x, context),
        }
        return branches

    def _score(self, branches: Mapping[str, Mapping[str, Tensor]], context: Tensor) -> Dict[str, object]:
        tokens = torch.stack([
            adapter(branches[name]["z"], branches[name]["raw_error"], branches[name]["evidence_logit"], context)
            for adapter, name in zip(self.evidence_adapters, self.branch_names)
        ], dim=2)
        evidence_logits = torch.stack([branches[name]["evidence_logit"] for name in self.branch_names], dim=2)
        score = self.joint_scorer(tokens, evidence_logits, context)
        score["tokens"] = tokens
        score["evidence_logits"] = evidence_logits
        return score

    def forward(
        self,
        x: Tensor,
        intervention: Optional[Mapping[str, Tensor]] = None,
        compute_loss: bool = True,
    ) -> Dict[str, object]:
        if x.ndim != 3 or x.size(-1) != self.config.c_in or x.size(1) != self.config.seq_len:
            raise ValueError(f"Expected [B,{self.config.seq_len},{self.config.c_in}], got {tuple(x.shape)}")
        with torch.no_grad():
            anchor = self.anchor(x)
        context = self.anchor_context_encoder(anchor["anchor_reconstruction"])
        branches = self._type_forward(x, context)
        scored = self._score(branches, context)
        output: Dict[str, object] = {
            **anchor,
            "anchor_context": context,
            "branches": branches,
            **scored,
            "joint_score": scored["joint_score"],
            "total_score": scored["joint_score"],
            # Intervention targets are consumed only by the training loss and
            # are never exposed as formal model predictions.
            "intervention": None,
            "intervention_output": None,
        }
        if intervention is not None:
            corrupted = intervention["corrupted_x"]
            int_branches = self._type_forward(corrupted, context)
            int_scored = self._score(int_branches, context)
            if "weak_view_i" in intervention:
                weak_i = self._score(self._type_forward(intervention["weak_view_i"], context), context)
                weak_j = self._score(self._type_forward(intervention["weak_view_j"], context), context)
                weak_compound = int_scored
                weak_type_mask = intervention["type_masks"].any(dim=-1)
                weak_mask = intervention.get("scenario_kind", torch.zeros(x.size(0), device=x.device)).eq(3).view(-1, 1)
                target_indices = intervention["type_targets"].topk(2, dim=1).indices
                mask_i = weak_type_mask.gather(1, target_indices[:, :1]).squeeze(1)
                mask_j = weak_type_mask.gather(1, target_indices[:, 1:2]).squeeze(1)
                int_scored["weak_views"] = {
                    "score_i": weak_i["joint_score"],
                    "score_j": weak_j["joint_score"],
                    "score_compound": weak_compound["joint_score"],
                    "mask_i": (mask_i * weak_mask).to(x.dtype),
                    "mask_j": (mask_j * weak_mask).to(x.dtype),
                    "union_mask": (intervention["union_mask"] * weak_mask).to(x.dtype),
                }
            output["intervention_output"] = {
                "branches": int_branches,
                "type_targets": intervention["type_targets"],
                "type_masks": intervention["type_masks"],
                "union_mask": intervention["union_mask"],
                **int_scored,
            }
        if compute_loss:
            output["losses"] = compute_phase_b_losses(output, output["intervention_output"], self.config)
        return output


def __getattr__(name: str):
    # Keep the requested ``typefusion_catch_v2.TypeFusionCATCHV2`` import path
    # available without creating a module-import cycle with the adapter.
    if name == "TypeFusionCATCHV2":
        from .TypeFusionCATCHV2 import TypeFusionCATCHV2
        return TypeFusionCATCHV2
    raise AttributeError(name)

"""End-to-end TypeFusion-CATCH v2 model without a CATCH anchor."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn

from .config import TypeFusionCATCHV2Config
from .layers import (
    EvidenceAdapterBank,
    EvolutionNormalityBranch,
    PatternNormalityBranch,
    RelationAwareJointScorer,
    RelationNormalityBranch,
    SharedRepresentationFrontend,
    StateNormalityBranch,
    TypeInterventionGenerator,
)
from .losses import compute_losses


class TypeFusionCATCHV2Model(nn.Module):
    branch_names = ("state", "evolution", "pattern", "relation")

    def __init__(self, config: TypeFusionCATCHV2Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.shared_frontend = SharedRepresentationFrontend(config)
        self.state_branch = StateNormalityBranch(config)
        self.evolution_branch = EvolutionNormalityBranch(config.c_in, config.branch_dim, config.temporal_layers, config.dropout)
        self.pattern_branch = PatternNormalityBranch(config)
        self.relation_branch = RelationNormalityBranch(config)
        self.evidence_adapters = EvidenceAdapterBank(config)
        self.joint_scorer = RelationAwareJointScorer(config)
        self.intervention_generator = TypeInterventionGenerator(config.seed)

    def _run_views(self, x: Tensor) -> Dict[str, object]:
        # The unmasked shared path is time-only.  Pattern explicitly owns two
        # masked frequency encodings inside its branch call.
        shared = self.shared_frontend.encode_time(x)
        branches = {
            "state": self.state_branch(shared["h_time"]),
            "evolution": self.evolution_branch(x),
            "pattern": self.pattern_branch(x, frontend=self.shared_frontend),
            "relation": self.relation_branch(x),
        }
        all_valid = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        branch_valid_mask = torch.stack((
            all_valid,
            branches["evolution"]["valid_mask"],
            all_valid,
            all_valid,
        ), dim=2)
        tokens = self.evidence_adapters(branches.values())
        # Adapter linear biases and type embeddings must not recreate an
        # invalid Evolution token at t=0.
        tokens = tokens * branch_valid_mask.unsqueeze(-1).to(tokens.dtype)
        evidence_logits = torch.stack([branches[name]["evidence_logit"] for name in self.branch_names], dim=2)
        scored = self.joint_scorer(tokens, evidence_logits, branch_valid_mask=branch_valid_mask)
        return {
            "shared": shared,
            "branches": branches,
            "tokens": tokens,
            "evidence_logits": evidence_logits,
            "branch_valid_mask": branch_valid_mask,
            **scored,
        }

    def forward(self, x: Tensor, intervention: Optional[Mapping[str, Tensor]] = None, compute_loss: bool = True) -> Dict[str, object]:
        if x.ndim != 3 or x.size(1) != self.config.seq_len or x.size(2) != self.config.c_in:
            raise ValueError(f"Expected [B,{self.config.seq_len},{self.config.c_in}], got {tuple(x.shape)}")
        clean = self._run_views(x)
        output: Dict[str, object] = {
            **clean["shared"],
            "branches": clean["branches"],
            "tokens": clean["tokens"],
            "evidence_logits": clean["evidence_logits"],
            "branch_valid_mask": clean["branch_valid_mask"],
            "joint_logit": clean["joint_logit"],
            "joint_score": clean["joint_score"],
            "total_score": clean["joint_score"],
            "intervention_output": None,
        }
        if intervention is not None:
            corrupted = self._run_views(intervention["corrupted_x"])
            intervention_output: Dict[str, object] = {
                **corrupted,
                "type_targets": intervention["type_targets"],
                "type_masks": intervention["type_masks"],
                "union_mask": intervention["union_mask"],
                "scenario_kind": intervention["scenario_kind"],
            }
            for debug_key in ("donor_indices", "debug_donor_indices", "debug_intervals", "debug_channels", "debug_strengths"):
                if debug_key in intervention:
                    intervention_output[debug_key] = intervention[debug_key]
            if intervention["scenario_kind"].eq(3).any():
                weak_i = self._run_views(intervention["weak_view_i"])
                weak_j = self._run_views(intervention["weak_view_j"])
                intervention_output["weak_views"] = {
                    "logit_i": weak_i["joint_logit"],
                    "logit_j": weak_j["joint_logit"],
                    "compound_logit": corrupted["joint_logit"],
                    "mask_i": intervention["weak_mask_i"],
                    "mask_j": intervention["weak_mask_j"],
                    "union_mask": intervention["union_mask"],
                }
            output["intervention_output"] = intervention_output
        if compute_loss:
            output["losses"] = compute_losses(output, output["intervention_output"], self.config)
        return output


def __getattr__(name: str):
    if name == "TypeFusionCATCHV2":
        from .TypeFusionCATCHV2 import TypeFusionCATCHV2
        return TypeFusionCATCHV2
    raise AttributeError(name)

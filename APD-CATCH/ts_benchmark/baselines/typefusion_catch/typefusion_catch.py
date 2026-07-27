"""PyTorch implementation of TypeFusion-CATCH v1."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.layers.branch_fusion_transformer import BranchFusionTransformer
from ts_benchmark.baselines.typefusion_catch.layers.catch_pattern_branch import CATCHPatternBranch
from ts_benchmark.baselines.typefusion_catch.layers.causal_evolution_branch import CausalEvolutionBranch
from ts_benchmark.baselines.typefusion_catch.layers.evidence_adapter import EvidenceAdapter
from ts_benchmark.baselines.typefusion_catch.layers.joint_normal_decoder import JointNormalDecoder
from ts_benchmark.baselines.typefusion_catch.layers.losses import compute_losses
from ts_benchmark.baselines.typefusion_catch.layers.masked_relation_branch import MaskedRelationBranch
from ts_benchmark.baselines.typefusion_catch.layers.memory_state_branch import MemoryStateBranch
from ts_benchmark.baselines.typefusion_catch.layers.shared_catch_stem import SharedCatchStem


class TypeFusionCATCHModel(nn.Module):
    """Four specialised normality branches with conflict-aware token fusion.

    The decoder's only data-dependent input is the four leave-one-out normal
    token predictions.  Branch evidence maps are diagnostics, not a score
    ensemble or a routing mechanism.
    """

    branch_names = ("state", "evolution", "pattern", "relation")

    def __init__(self, config: TypeFusionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.shared_stem = SharedCatchStem(config)
        self.state_branch = MemoryStateBranch(config)
        self.evolution_branch = CausalEvolutionBranch(config)
        self.pattern_branch = CATCHPatternBranch(config)
        self.relation_branch = MaskedRelationBranch(config)
        self.evidence_adapters = nn.ModuleDict(
            {name: EvidenceAdapter(config, index) for index, name in enumerate(self.branch_names)}
        )
        self.branch_fusion = BranchFusionTransformer(config)
        self.joint_decoder = JointNormalDecoder(config)
        self.set_training_stage(config.training_stage)

    @staticmethod
    def _set_module_trainable(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = enabled

    def set_training_stage(self, training_stage: str) -> None:
        """Apply the documented three-stage optimisation boundary."""

        if training_stage not in {"branch_pretrain", "fusion_train", "joint_finetune"}:
            raise ValueError(f"Unsupported training stage: {training_stage}")
        self.config.training_stage = training_stage
        for module in self.children():
            self._set_module_trainable(module, False)

        if training_stage == "branch_pretrain":
            modules = (
                self.shared_stem,
                self.state_branch,
                self.evolution_branch,
                self.pattern_branch,
                self.relation_branch,
            )
            for module in modules:
                self._set_module_trainable(module, True)
        elif training_stage == "fusion_train":
            for module in (self.evidence_adapters, self.branch_fusion, self.joint_decoder):
                self._set_module_trainable(module, True)
        else:
            for module in (self.evidence_adapters, self.branch_fusion, self.joint_decoder):
                self._set_module_trainable(module, True)
            # Only late shared projections and branch output layers are unfrozen.
            for module in (
                self.shared_stem.temporal_projection,
                self.shared_stem.channel_fusion.output_projection,
            ):
                self._set_module_trainable(module, True)
            for branch in (
                self.state_branch,
                self.evolution_branch,
                self.pattern_branch,
                self.relation_branch,
            ):
                branch.unfreeze_tail()

    def train(self, mode: bool = True) -> "TypeFusionCATCHModel":
        """Keep frozen branch computations deterministic in fusion stages.

        Freezing a parameter alone does not freeze BatchNorm buffers or disable
        dropout.  Tail layers still receive gradients in joint fine-tuning while
        their frozen upstream branch computations remain in evaluation mode.
        """

        super().train(mode)
        if mode and self.config.training_stage in {"fusion_train", "joint_finetune"}:
            for module in (
                self.shared_stem,
                self.state_branch,
                self.evolution_branch,
                self.pattern_branch,
                self.relation_branch,
            ):
                module.eval()
        return self

    def _adapt_evidence(self, branches: Dict[str, Dict[str, Tensor]]) -> Tensor:
        branch_tokens = [
            self.evidence_adapters[name](
                branches[name]["z"], branches[name]["e"], branches[name]["x_hat"]
            )
            for name in self.branch_names
        ]
        return torch.stack(branch_tokens, dim=2)

    def forward(self, x: Tensor, compute_joint: bool | None = None) -> Dict[str, object]:
        if compute_joint is None:
            compute_joint = self.config.training_stage != "branch_pretrain"
        if not compute_joint and self.config.training_stage != "branch_pretrain":
            raise ValueError("compute_joint=False is only valid during branch_pretrain")
        shared = self.shared_stem(x)
        normalized_input = shared["normalized_input"]
        branches = {
            "state": self.state_branch(normalized_input, shared["temporal_latent"]),
            # x has already passed the train-data StandardScaler in the adapter.
            # It intentionally bypasses full-window RevIN statistics so the
            # causal branch has no target/future normalisation leakage.
            "evolution": self.evolution_branch(x),
            "pattern": self.pattern_branch(
                normalized_input,
                shared["frequency_channels"],
                training_mask=self.training and self.config.training_stage != "fusion_train",
            ),
            "relation": self.relation_branch(
                normalized_input,
                randomize_groups=self.training and self.config.training_stage != "fusion_train",
            ),
        }
        if not compute_joint:
            output: Dict[str, object] = {
                "normalized_input": normalized_input,
                "evolution_input": x,
                "spectrum": shared["spectrum"],
                "branches": branches,
                "q": None,
                "q_normal": None,
                "branch_mask_prediction": None,
                "branch_mask_loss": normalized_input.new_zeros(()),
                "leave_one_out": None,
                "x_hat_joint_normalized": None,
                "x_hat_joint": None,
                "total_score": None,
                "branch_conflict_map": None,
            }
            output["losses"] = compute_losses(output, self.config)
            return output

        q = self._adapt_evidence(branches)
        leave_one_out = self.branch_fusion.leave_one_out(q)
        if self.training and self.config.training_stage != "branch_pretrain":
            mask_prediction = self.branch_fusion.masked_branch_prediction(q)
            branch_mask_loss = mask_prediction["loss"]
        elif not self.training and self.config.training_stage != "branch_pretrain":
            mask_prediction = None
            branch_mask_loss = self.branch_fusion.leave_one_out_mask_loss(q, leave_one_out)
        else:
            mask_prediction = None
            branch_mask_loss = q.new_zeros(())

        q_normal = leave_one_out["q_normal"]
        x_hat_joint_normalized = self.joint_decoder(q_normal)
        x_hat_joint = self.shared_stem.revin.denormalize(
            x_hat_joint_normalized, shared["revin_statistics"]
        )
        output: Dict[str, object] = {
            "normalized_input": normalized_input,
            "evolution_input": x,
            "spectrum": shared["spectrum"],
            "branches": branches,
            "q": q,
            "q_normal": q_normal,
            "branch_mask_prediction": mask_prediction,
            "branch_mask_loss": branch_mask_loss,
            "leave_one_out": leave_one_out,
            "x_hat_joint_normalized": x_hat_joint_normalized,
            "x_hat_joint": x_hat_joint,
            "total_score": (x - x_hat_joint).abs(),
            "branch_conflict_map": (q - q_normal).abs().mean(dim=-1),
        }
        output["losses"] = compute_losses(output, self.config)
        return output

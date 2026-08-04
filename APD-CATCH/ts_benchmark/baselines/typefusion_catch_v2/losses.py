"""Finite-checked single-stage v2 objectives."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from .config import TypeFusionCATCHV2Config


def _check(name: str, value: Tensor) -> Tensor:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"non-finite loss component: {name}")
    return value


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def compute_losses(clean: Mapping[str, object], intervention: Optional[Mapping[str, object]], config: TypeFusionCATCHV2Config) -> Dict[str, Tensor]:
    branches = clean["branches"]
    task = sum(branch["task_loss"] for branch in branches.values())
    clean_evidence = torch.stack([
        F.binary_cross_entropy_with_logits(branch["evidence_logit"], torch.zeros_like(branch["evidence_logit"]))
        for branch in branches.values()
    ]).mean()
    clean_score = clean["joint_score"].mean()
    target_positive = clean["joint_logit"].new_zeros(())
    responsibility = clean["joint_logit"].new_zeros(())
    score = clean["joint_logit"].new_zeros(())
    score_rank = clean["joint_logit"].new_zeros(())
    synergy = clean["joint_logit"].new_zeros(())

    if intervention is not None:
        targets = intervention["type_targets"].to(clean["joint_logit"].device).float()
        masks = intervention["type_masks"].to(clean["joint_logit"].device).float().amax(dim=-1)
        union = intervention["union_mask"].to(clean["joint_logit"].device).float()
        int_branches = intervention["branches"]
        logits = torch.stack([int_branches[name]["evidence_logit"] for name in ("state", "evolution", "pattern", "relation")], dim=1)
        positive_terms = []
        responses = torch.sigmoid(logits)
        for index in range(4):
            selected = targets[:, index] > 0.5
            if selected.any():
                positive_terms.append(F.binary_cross_entropy_with_logits(logits[:, index][selected], masks[:, index][selected]))
        if positive_terms:
            target_positive = torch.stack(positive_terms).mean()
        for index in range(4):
            target_i = targets[:, index] > 0.5
            for other in range(4):
                target_other = targets[:, other] > 0.5
                selected = target_i & ~target_other
                if selected.any():
                    target_response = _masked_mean(responses[:, index][selected], masks[:, index][selected])
                    other_response = _masked_mean(responses[:, other][selected], masks[:, index][selected])
                    responsibility = responsibility + F.relu(config.responsibility_margin - target_response + other_response)
        joint_logit = intervention["joint_logit"]
        positive_weight = ((union.numel() - union.sum()) / union.sum().clamp_min(1.0)).clamp(1.0, 20.0).detach()
        score = F.binary_cross_entropy_with_logits(joint_logit, union, pos_weight=positive_weight)
        joint_score = F.softplus(joint_logit)
        valid_regions = (union.sum(dim=1) > 0) & ((1.0 - union).sum(dim=1) > 0)
        if valid_regions.any():
            positive_mean = (joint_score * union).sum(dim=1) / union.sum(dim=1).clamp_min(1.0)
            negative = 1.0 - union
            negative_mean = (joint_score * negative).sum(dim=1) / negative.sum(dim=1).clamp_min(1.0)
            score_rank = F.relu(config.score_margin - positive_mean + negative_mean)[valid_regions].mean()
        weak_mask = intervention.get("scenario_kind", torch.zeros(targets.size(0), device=targets.device)).eq(3)
        weak_views = intervention.get("weak_views")
        if weak_views is not None and weak_mask.any():
            component_i = F.softplus(weak_views["logit_i"])
            component_j = F.softplus(weak_views["logit_j"])
            compound = F.softplus(weak_views["compound_logit"])
            score_i = _masked_mean(component_i[weak_mask], weak_views["mask_i"][weak_mask])
            score_j = _masked_mean(component_j[weak_mask], weak_views["mask_j"][weak_mask])
            score_compound = _masked_mean(compound[weak_mask], weak_views["union_mask"][weak_mask])
            synergy = F.relu(config.synergy_margin - score_compound + torch.maximum(score_i, score_j))

    evidence = clean_evidence + target_positive
    components = {
        "task": task,
        "evidence": evidence,
        "clean_evidence": clean_evidence,
        "target_positive": target_positive,
        "responsibility": responsibility,
        "score": score,
        "score_rank": score_rank,
        "clean_score": clean_score,
        "synergy": synergy,
    }
    total = (
        config.lambda_task * task + config.lambda_evidence * evidence
        + config.lambda_responsibility * responsibility + config.lambda_score * score
        + config.lambda_score_rank * score_rank + config.lambda_clean_score * clean_score
        + config.lambda_synergy * synergy
    )
    components["total"] = total
    return {name: _check(name, value) for name, value in components.items()}


compute_phase_b_losses = compute_losses

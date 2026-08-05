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


def _branch_valid_mask(branches: Mapping[str, object], reference: Tensor) -> Tensor:
    masks = []
    for name in ("state", "evolution", "pattern", "relation"):
        branch = branches[name]
        valid = branch.get("valid_mask") if isinstance(branch, Mapping) else None
        if valid is None:
            valid = torch.ones_like(reference, dtype=torch.bool)
        masks.append(valid.to(device=reference.device, dtype=torch.bool))
    return torch.stack(masks, dim=1)


def compute_losses(clean: Mapping[str, object], intervention: Optional[Mapping[str, object]], config: TypeFusionCATCHV2Config) -> Dict[str, Tensor]:
    branches = clean["branches"]
    task = sum(branch["task_loss"] for branch in branches.values())
    clean_valid = clean.get("branch_valid_mask")
    if clean_valid is None:
        clean_valid = _branch_valid_mask(branches, clean["joint_logit"])
    clean_evidence_terms = []
    for index, name in enumerate(("state", "evolution", "pattern", "relation")):
        logits = branches[name]["evidence_logit"]
        loss_map = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none")
        clean_evidence_terms.append(_masked_mean(loss_map, clean_valid[:, :, index]))
    clean_evidence = torch.stack(clean_evidence_terms).mean()
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
        branch_valid_mask = intervention.get("branch_valid_mask")
        if branch_valid_mask is None:
            branch_valid_mask = _branch_valid_mask(int_branches, clean["joint_logit"])
        else:
            branch_valid_mask = branch_valid_mask.to(device=clean["joint_logit"].device, dtype=torch.bool)
        logits = torch.stack([int_branches[name]["evidence_logit"] for name in ("state", "evolution", "pattern", "relation")], dim=1)
        positive_terms = []
        responses = torch.sigmoid(logits)
        for index in range(4):
            selected = targets[:, index] > 0.5
            if selected.any():
                loss_map = F.binary_cross_entropy_with_logits(logits[:, index], masks[:, index], reduction="none")
                effective_valid = branch_valid_mask[:, :, index] & selected.unsqueeze(1)
                positive_terms.append(_masked_mean(loss_map, effective_valid))
        if positive_terms:
            target_positive = torch.stack(positive_terms).mean()
        relation_terms = []
        relation_valid = []
        for index in range(4):
            target_i = targets[:, index] > 0.5
            for other in range(4):
                # A compound's two active types are both targets and therefore
                # never form a responsibility comparison against each other.
                target_other = targets[:, other] > 0.5
                effective_region = masks[:, index] * branch_valid_mask[:, :, index].to(masks.dtype) * branch_valid_mask[:, :, other].to(masks.dtype)
                target_count = effective_region.sum(dim=1).clamp_min(1.0)
                target_response = (responses[:, index] * effective_region).sum(dim=1) / target_count
                valid_pair = target_i & ~target_other & (effective_region.sum(dim=1) > 0)
                other_response = (responses[:, other] * effective_region).sum(dim=1) / target_count
                relation_terms.append(F.relu(config.responsibility_margin - target_response + other_response))
                relation_valid.append(valid_pair)
        if relation_terms:
            relation_values = torch.stack(relation_terms, dim=1)
            relation_mask = torch.stack(relation_valid, dim=1)
            responsibility = relation_values[relation_mask].mean() if relation_mask.any() else relation_values.sum() * 0.0
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
            mask_i = weak_views["mask_i"].to(component_i.dtype)
            mask_j = weak_views["mask_j"].to(component_j.dtype)
            mask_union = weak_views["union_mask"].to(compound.dtype)
            count_i = mask_i.sum(dim=1).clamp_min(1.0)
            count_j = mask_j.sum(dim=1).clamp_min(1.0)
            count_union = mask_union.sum(dim=1).clamp_min(1.0)
            score_i = (component_i * mask_i).sum(dim=1) / count_i
            score_j = (component_j * mask_j).sum(dim=1) / count_j
            score_compound = (compound * mask_union).sum(dim=1) / count_union
            valid_weak = weak_mask & (mask_i.sum(dim=1) > 0) & (mask_j.sum(dim=1) > 0) & (mask_union.sum(dim=1) > 0)
            synergy_values = F.relu(config.synergy_margin - score_compound + torch.maximum(score_i, score_j))
            synergy = synergy_values[valid_weak].mean() if valid_weak.any() else synergy_values.sum() * 0.0

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

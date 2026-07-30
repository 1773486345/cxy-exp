"""Fixed v2 self-supervised type and joint-score objectives."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from .config import TypeFusionCATCHV2Config


def _finite(value: Tensor) -> Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(value.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


def compute_phase_b_losses(
    clean: Mapping[str, object],
    intervention: Optional[Mapping[str, object]],
    config: TypeFusionCATCHV2Config,
) -> Dict[str, Tensor]:
    """Compute every Phase B term from clean and optional intervention views."""

    reference = clean["branches"]
    task_terms = [branch.get("task_loss", clean["joint_logit"].new_zeros(())) for branch in reference.values()]
    task_loss = _finite(torch.stack([item.reshape(()) for item in task_terms]).sum())

    evidence_terms = []
    for branch in reference.values():
        evidence_terms.append(F.binary_cross_entropy_with_logits(
            _finite(branch["evidence_logit"]), torch.zeros_like(branch["evidence_logit"])
        ))
    clean_evidence = torch.stack(evidence_terms).mean()

    clean_score = _finite(clean["joint_score"]).mean()
    responsibility = clean["joint_logit"].new_zeros(())
    target_positive = clean["joint_logit"].new_zeros(())
    score_loss = clean["joint_logit"].new_zeros(())
    score_rank = clean["joint_logit"].new_zeros(())
    synergy = clean["joint_logit"].new_zeros(())

    if intervention is not None:
        targets = intervention["type_targets"].to(clean["joint_logit"].device).float()
        masks = intervention["type_masks"].to(clean["joint_logit"].device).float()
        union = intervention["union_mask"].to(clean["joint_logit"].device).float()
        int_branches = intervention["branches"]
        responses = []
        for index, name in enumerate(("state", "evolution", "pattern", "relation")):
            logit = _finite(int_branches[name]["evidence_logit"])
            target = masks[:, index].amax(dim=-1)
            selected = (targets[:, index] > 0.5).nonzero(as_tuple=False).flatten()
            if selected.numel():
                target_positive = target_positive + F.binary_cross_entropy_with_logits(
                    logit.index_select(0, selected), target.index_select(0, selected)
                )
                responses.append(torch.sigmoid(logit).index_select(0, selected))
            else:
                responses.append(logit.new_zeros(1))
        # A target type is encouraged to exceed every non-target type, but the
        # two members of a compound are never made mutually exclusive.
        response_matrix = torch.stack([
            torch.sigmoid(_finite(int_branches[name]["evidence_logit"])) for name in ("state", "evolution", "pattern", "relation")
        ], dim=1)
        for i in range(4):
            target_i = targets[:, i].view(-1, 1, 1)
            response_i = response_matrix[:, i]
            for j in range(4):
                if i == j:
                    continue
                response_j = response_matrix[:, j]
                pair_mask = (target_i.squeeze(-1).squeeze(-1) > 0.5)
                responsibility = responsibility + F.relu(
                    config.responsibility_margin - response_i + response_j
                ).index_select(0, pair_mask.nonzero(as_tuple=False).flatten()).mean() if bool(pair_mask.any()) else responsibility

        joint_logit = _finite(intervention["joint_logit"])
        positive_weight = ((union.numel() - union.sum()) / union.sum().clamp_min(1.0)).clamp(1.0, 20.0)
        score_loss = F.binary_cross_entropy_with_logits(
            joint_logit, union, pos_weight=positive_weight.detach()
        )
        joint_score = F.softplus(joint_logit)
        score_rank = F.relu(
            config.score_margin
            - _masked_mean(joint_score, union)
            + _masked_mean(joint_score, 1.0 - union)
        )
        weak_views = intervention.get("weak_views")
        if weak_views:
            weak_i = _masked_mean(F.softplus(weak_views["score_i"]), weak_views["mask_i"])
            weak_j = _masked_mean(F.softplus(weak_views["score_j"]), weak_views["mask_j"])
            weak_c = _masked_mean(F.softplus(weak_views["score_compound"]), weak_views["union_mask"])
            synergy = F.relu(config.synergy_margin - weak_c + torch.maximum(weak_i, weak_j))

    evidence_loss = clean_evidence + target_positive
    total = (
        config.lambda_task * task_loss
        + config.lambda_evidence * evidence_loss
        + config.lambda_responsibility * responsibility
        + config.lambda_score * score_loss
        + config.lambda_score_rank * score_rank
        + config.lambda_clean_score * clean_score
        + config.lambda_synergy * synergy
    )
    values = {
        "total": total,
        "task": task_loss,
        "evidence": evidence_loss,
        "clean_evidence": clean_evidence,
        "target_positive": target_positive,
        "responsibility": responsibility,
        "score": score_loss,
        "score_rank": score_rank,
        "clean_score": clean_score,
        "synergy": synergy,
    }
    return {name: _finite(value) for name, value in values.items()}


def compute_losses(clean: Mapping[str, object], intervention: Optional[Mapping[str, object]], config: TypeFusionCATCHV2Config) -> Dict[str, Tensor]:
    return compute_phase_b_losses(clean, intervention, config)

"""Fixed, interpretable TypeFusion-CATCH training objectives."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
import torch.nn.functional as F

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig


def huber(prediction: Tensor, target: Tensor) -> Tensor:
    return F.smooth_l1_loss(prediction, target)


def compute_losses(output: Dict[str, object], config: TypeFusionConfig) -> Dict[str, Tensor]:
    normalized_input = output["normalized_input"]
    evolution_input = output["evolution_input"]
    branches = output["branches"]
    assert isinstance(normalized_input, Tensor)
    assert isinstance(evolution_input, Tensor)
    assert isinstance(branches, dict)
    state = branches["state"]
    evolution = branches["evolution"]
    pattern = branches["pattern"]
    relation = branches["relation"]

    losses = {
        "state": huber(state["x_hat"], normalized_input),
        "evolution": huber(evolution["x_hat"], evolution_input),
        "pattern_time": huber(pattern["x_hat"], normalized_input),
        "relation": huber(relation["x_hat"], normalized_input),
        "pattern_freq": F.mse_loss(pattern["predicted_spectrum"].real, output["spectrum"].real)
        + F.mse_loss(pattern["predicted_spectrum"].imag, output["spectrum"].imag),
        "branch_mask": output["branch_mask_loss"],
        "joint": huber(output["x_hat_joint_normalized"], normalized_input),
    }
    branch_total = losses["state"] + losses["evolution"] + losses["pattern_time"] + losses["relation"]
    branch_total = branch_total + config.lambda_freq * losses["pattern_freq"]
    fusion_total = config.lambda_mask * losses["branch_mask"] + losses["joint"]
    if config.training_stage == "branch_pretrain":
        total = branch_total
    elif config.training_stage == "fusion_train":
        total = fusion_total
    else:
        total = branch_total + fusion_total
    losses["total"] = total
    return losses

"""Strict CATCH-common checkpoint reconstruction helpers for the adapter."""

from ts_benchmark.baselines.catch.CATCH import TransformerConfig
from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel


def load_verified_catch_checkpoint(payload, evidence):
    if not isinstance(payload, dict) or payload.get("protocol") != "CATCH-common":
        raise ValueError("checkpoint is not a CATCH-common artifact")
    config_values = payload.get("model_config")
    if not isinstance(config_values, dict) or evidence.get("model_config") != config_values:
        raise ValueError("checkpoint model configuration is missing or differs from execution evidence")
    if payload.get("score_protocol") != evidence.get("score_protocol"):
        raise ValueError("checkpoint score protocol differs from execution evidence")
    if (payload.get("normalization_sha256") != evidence.get("normalization_sha256")
            or payload.get("train_fit_rows_sha256") != evidence.get("train_fit_rows_sha256")):
        raise ValueError("checkpoint normalization or training-row binding differs from evidence")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint state_dict is missing")
    config = TransformerConfig(**config_values)
    if int(config.seq_len) != int(evidence.get("seq_len", -1)):
        raise ValueError("checkpoint sequence length differs from its execution evidence")
    model = CATCHModel(config)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as error:
        raise ValueError(f"checkpoint does not strictly load into the recorded CATCH model: {error}") from error
    return model

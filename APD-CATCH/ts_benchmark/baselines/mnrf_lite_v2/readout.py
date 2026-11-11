"""Score readouts on the fixed S coordinate and the existing Flow score."""

import numpy as np


SCORE_READOUTS = ("flow", "state_amplitude", "flow_plus_amplitude")


def validate_score_readout(readout):
    if readout not in SCORE_READOUTS:
        raise ValueError(f"score_readout must be one of {SCORE_READOUTS}, got {readout!r}")
    return readout


def score_target_indices(variable_names, target=""):
    """Resolve once by the frozen name/order, shared by calibration and testing."""
    names = list(variable_names)
    if not target:
        return None
    if names.count(target) != 1:
        raise ValueError(f"score target {target!r} must occur exactly once in variable_names")
    return [names.index(target)]


def flow_scores(nll, center, scale, score_indices=None):
    standardized = (nll - center) / scale
    if score_indices is not None:
        standardized = standardized[:, score_indices]
    return np.max(standardized, axis=1)


def state_amplitude_scores(features, state_index, score_indices=None):
    state = features[..., state_index]
    if score_indices is not None:
        state = state[:, score_indices]
    return np.max(np.abs(state), axis=1)


def combine_scores(flow, amplitude, calibration):
    center, scale = calibration["center"], calibration["scale"]
    return np.maximum((flow - center[0]) / scale[0],
                      (amplitude - center[1]) / scale[1])


def fit_amplitude_calibration(validation, calibration, quantile, scale_floor):
    """Columns are aggregate Flow score and input-standardized S amplitude."""
    center = np.median(validation, axis=0)
    mad = np.median(np.abs(validation - center), axis=0)
    state = {"center": center.tolist(), "scale": np.maximum(1.4826 * mad, scale_floor).tolist()}
    state["thresholds"] = {
        "state_amplitude": float(np.quantile(calibration[:, 1], quantile)),
        "flow_plus_amplitude": float(np.quantile(
            combine_scores(calibration[:, 0], calibration[:, 1], state), quantile
        )),
    }
    return validate_amplitude_calibration(state)


def validate_amplitude_calibration(state):
    try:
        center = np.asarray(state["center"], dtype=np.float64)
        scale = np.asarray(state["scale"], dtype=np.float64)
        thresholds = {name: float(state["thresholds"][name]) for name in SCORE_READOUTS[1:]}
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("invalid state amplitude readout calibration") from error
    if (center.shape != (2,) or scale.shape != (2,) or not np.isfinite(center).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.isfinite(list(thresholds.values())).all()):
        raise ValueError("invalid state amplitude readout calibration")
    return {"center": center.tolist(), "scale": scale.tolist(), "thresholds": thresholds}

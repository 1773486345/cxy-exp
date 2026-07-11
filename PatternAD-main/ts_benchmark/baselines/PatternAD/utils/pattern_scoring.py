import math

import numpy as np
from scipy.special import gammaln, log_ndtr
from scipy.stats import t as student_t_distribution


class PatternAwareScorer:
    """Context-aware reconstruction scorer for multivariate TSAD.

    The first PatternAD scorer treated raw, scale, trend, frequency, and
    cross-variable scores as independent anomaly evidence, then aggregated them.
    The raw-control experiment showed that this often hurts ranking quality.

    The default mode now returns raw reconstruction residual. If conditional
    scoring is enabled by the model, the residual is computed only on positions
    that were hidden from the reconstructor. Context should improve the
    reconstruction itself, not multiply the final score after the fact.
    """

    DEFAULT_COMPONENTS = ("raw", "scale", "trend", "shift", "freq", "sync")
    CONTEXT_NAMES = ("scale_context", "trend_context", "freq_context", "sync_context")

    def __init__(
        self,
        components=None,
        local_window=5,
        trend_window=7,
        aggregation="mean",
        top_k=2,
        logsumexp_tau=1.0,
        eps=1e-6,
        use_calibration=False,
        score_mode="raw",
        context_strength=0.35,
        risk_strength=0.15,
        min_weight=0.5,
        max_weight=1.5,
        distribution="mse",
    ):
        self.components = tuple(components or self.DEFAULT_COMPONENTS)
        self.local_window = max(3, int(local_window))
        self.trend_window = max(3, int(trend_window))
        self.aggregation = aggregation
        self.top_k = max(1, int(top_k))
        self.logsumexp_tau = float(logsumexp_tau)
        self.eps = float(eps)
        self.use_calibration = bool(use_calibration)
        self.score_mode = str(score_mode)
        self.context_strength = float(context_strength)
        self.risk_strength = float(risk_strength)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        distribution = str(distribution).lower().replace("-", "_")
        aliases = {
            "raw": "mse",
            "normal": "gaussian",
            "student": "student_t",
            "studentt": "student_t",
        }
        self.distribution = aliases.get(distribution, distribution)
        if self.distribution not in {"mse", "gaussian", "student_t"}:
            raise ValueError("distribution must be one of: mse, gaussian, student_t.")
        self.stats = {}
        self.fitted = False

    @staticmethod
    def _as_numpy(value):
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else value
        array = np.asarray(array, dtype=np.float64)
        if array.ndim != 3:
            raise ValueError("PatternAwareScorer expects windows shaped [B, T, D].")
        return array

    @staticmethod
    def _odd_window(window):
        window = max(3, int(window))
        return window + 1 if window % 2 == 0 else window

    @staticmethod
    def _robust_stat(value, eps):
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        center = float(np.median(flat))
        scale = float(np.median(np.abs(flat - center)) * 1.4826)
        if scale < eps:
            scale = float(np.std(flat))
        if scale < eps:
            scale = 1.0
        return {"center": center, "scale": scale}

    def _rolling_mean(self, x, window):
        window = self._odd_window(window)
        radius = window // 2
        padded = np.pad(x, ((0, 0), (radius, radius), (0, 0)), mode="edge")
        out = np.empty_like(x, dtype=np.float64)
        for t in range(x.shape[1]):
            out[:, t, :] = padded[:, t:t + window, :].mean(axis=1)
        return out

    def _rolling_std(self, x, window):
        mean = self._rolling_mean(x, window)
        mean_sq = self._rolling_mean(x * x, window)
        var = np.maximum(mean_sq - mean * mean, 0.0)
        return np.sqrt(var + self.eps)

    def _left_right_shift(self, x, window):
        window = max(2, int(window))
        radius = max(1, window // 2)
        padded = np.pad(x, ((0, 0), (radius, radius), (0, 0)), mode="edge")
        out = np.empty_like(x, dtype=np.float64)
        for t in range(x.shape[1]):
            left = padded[:, t:t + radius, :].mean(axis=1)
            right = padded[:, t + radius + 1:t + 2 * radius + 1, :].mean(axis=1)
            out[:, t, :] = np.abs(right - left)
        return out

    def _evidence_arrays(self, true, pred, score_mask=None):
        true = self._as_numpy(true)
        pred = self._as_numpy(pred)
        if true.shape != pred.shape:
            raise ValueError(
                "PatternAwareScorer true/pred shape mismatch: "
                f"{true.shape} vs {pred.shape}"
            )

        residual = (true - pred) ** 2
        mask = None
        if score_mask is not None:
            mask = score_mask.detach().cpu().numpy() if hasattr(score_mask, "detach") else score_mask
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != residual.shape:
                raise ValueError(
                    "PatternAwareScorer score_mask shape mismatch: "
                    f"{mask.shape} vs {residual.shape}"
                )
            denom = np.maximum(mask.sum(axis=-1), 1)
            raw = (residual * mask).sum(axis=-1) / denom
        else:
            raw = residual.mean(axis=-1)

        local_scale = self._rolling_std(true, self.local_window)
        if mask is not None:
            scale = ((residual / (local_scale ** 2 + self.eps)) * mask).sum(axis=-1) / denom
        else:
            scale = (residual / (local_scale ** 2 + self.eps)).mean(axis=-1)

        true_trend = self._rolling_mean(true, self.trend_window)
        pred_trend = self._rolling_mean(pred, self.trend_window)
        trend_residual = (true_trend - pred_trend) ** 2
        trend = (trend_residual * mask).sum(axis=-1) / denom if mask is not None else trend_residual.mean(axis=-1)

        true_shift = self._left_right_shift(true, self.trend_window)
        pred_shift = self._left_right_shift(pred, self.trend_window)
        shift_residual = (true_shift - pred_shift) ** 2
        shift = (shift_residual * mask).sum(axis=-1) / denom if mask is not None else shift_residual.mean(axis=-1)

        true_high = true - true_trend
        pred_high = pred - pred_trend
        freq_residual = (true_high - pred_high) ** 2
        freq = (freq_residual * mask).sum(axis=-1) / denom if mask is not None else freq_residual.mean(axis=-1)

        median = np.median(residual, axis=-1, keepdims=True)
        mad = np.median(np.abs(residual - median), axis=-1, keepdims=True)
        active = residual > (median + 1.4826 * mad + self.eps)
        active_ratio = active.mean(axis=-1)
        sync = raw * (1.0 + active_ratio)

        return {
            "raw": raw,
            "scale": scale,
            "trend": trend,
            "shift": shift,
            "freq": freq,
            "sync": sync,
            # Contexts describe the local operating state. They should not
            # become standalone anomaly scores in the default path.
            "scale_context": (local_scale ** 2).mean(axis=-1),
            "trend_context": true_shift.mean(axis=-1),
            "freq_context": (true_high ** 2).mean(axis=-1),
            "sync_context": active_ratio,
        }

    def component_scores(self, true, pred, score_mask=None):
        arrays = self._evidence_arrays(true, pred, score_mask=score_mask)
        return {name: arrays[name] for name in self.components if name in arrays}

    def fit(self, true_windows, pred_windows):
        arrays = self._evidence_arrays(true_windows, pred_windows)
        self.stats = {}
        if self.use_calibration:
            for name, value in arrays.items():
                self.stats[name] = self._robust_stat(value, self.eps)
        self.fitted = True
        return self

    def _distribution_nll(self, true_windows, pred_windows, distribution_params):
        true = self._as_numpy(true_windows)
        pred = self._as_numpy(pred_windows)
        if true.shape != pred.shape:
            raise ValueError(
                "PatternAwareScorer true/pred shape mismatch: "
                f"{true.shape} vs {pred.shape}"
            )
        if not distribution_params or "scale" not in distribution_params:
            raise ValueError(f"{self.distribution} NLL scoring requires scale parameters.")

        scale = self._as_numpy(distribution_params["scale"])
        if scale.shape != true.shape:
            raise ValueError(
                "PatternAwareScorer scale shape mismatch: "
                f"{scale.shape} vs {true.shape}"
            )
        if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
            raise ValueError("Distribution scale must be finite and strictly positive.")
        standardized = (true - pred) / np.maximum(scale, self.eps)
        if self.distribution == "gaussian":
            return (
                0.5 * standardized ** 2
                + np.log(scale)
                + 0.5 * math.log(2.0 * math.pi)
            )

        if "df" not in distribution_params:
            raise ValueError("student_t NLL scoring requires df parameters.")
        df = self._as_numpy(distribution_params["df"])
        if df.shape != true.shape:
            raise ValueError(
                "PatternAwareScorer df shape mismatch: "
                f"{df.shape} vs {true.shape}"
            )
        if np.any(df <= 2.0) or not np.all(np.isfinite(df)):
            raise ValueError("Student-t df must be finite and greater than 2.")
        return (
            gammaln(0.5 * df)
            - gammaln(0.5 * (df + 1.0))
            + 0.5 * (np.log(df) + math.log(math.pi))
            + np.log(scale)
            + 0.5 * (df + 1.0) * np.log1p(standardized ** 2 / df)
        )

    def _distribution_tail_surprisal(
        self, true_windows, pred_windows, distribution_params
    ):
        true = self._as_numpy(true_windows)
        pred = self._as_numpy(pred_windows)
        if true.shape != pred.shape:
            raise ValueError(
                "PatternAwareScorer true/pred shape mismatch: "
                f"{true.shape} vs {pred.shape}"
            )
        if not distribution_params or "scale" not in distribution_params:
            raise ValueError(
                f"{self.distribution} tail scoring requires scale parameters."
            )
        scale = self._as_numpy(distribution_params["scale"])
        if scale.shape != true.shape:
            raise ValueError(
                "PatternAwareScorer scale shape mismatch: "
                f"{scale.shape} vs {true.shape}"
            )
        if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
            raise ValueError("Distribution scale must be finite and strictly positive.")
        standardized = np.abs((true - pred) / np.maximum(scale, self.eps))
        if self.distribution == "gaussian":
            return -math.log(2.0) - log_ndtr(-standardized)

        if "df" not in distribution_params:
            raise ValueError("student_t tail scoring requires df parameters.")
        df = self._as_numpy(distribution_params["df"])
        if df.shape != true.shape:
            raise ValueError(
                "PatternAwareScorer df shape mismatch: "
                f"{df.shape} vs {true.shape}"
            )
        if np.any(df <= 2.0) or not np.all(np.isfinite(df)):
            raise ValueError("Student-t df must be finite and greater than 2.")
        return -math.log(2.0) - student_t_distribution.logsf(standardized, df)

    def _aggregate_cell_scores(self, cell_scores, score_mask=None):
        if score_mask is None:
            return cell_scores.mean(axis=-1)
        mask = score_mask.detach().cpu().numpy() if hasattr(score_mask, "detach") else score_mask
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != cell_scores.shape:
            raise ValueError(
                "PatternAwareScorer score_mask shape mismatch: "
                f"{mask.shape} vs {cell_scores.shape}"
            )
        denom = np.maximum(mask.sum(axis=-1), 1)
        return (cell_scores * mask).sum(axis=-1) / denom

    def _raw_residual_score(self, true_windows, pred_windows, score_mask=None):
        true = self._as_numpy(true_windows)
        pred = self._as_numpy(pred_windows)
        if true.shape != pred.shape:
            raise ValueError(
                "PatternAwareScorer true/pred shape mismatch: "
                f"{true.shape} vs {pred.shape}"
            )
        return self._aggregate_cell_scores(
            (true - pred) ** 2, score_mask=score_mask
        )

    def _z(self, name, value):
        stat = self.stats[name]
        return (value - stat["center"]) / (stat["scale"] + self.eps)

    def transform_components(self, true_windows, pred_windows):
        if not self.fitted:
            raise RuntimeError("PatternAwareScorer must be fitted before scoring.")
        scores = self.component_scores(true_windows, pred_windows)
        if not self.use_calibration:
            return scores
        calibrated = {}
        for name, value in scores.items():
            calibrated[name] = np.maximum(self._z(name, value), 0.0)
        return calibrated

    def _aggregate_components(self, true_windows, pred_windows):
        calibrated = self.transform_components(true_windows, pred_windows)
        if not calibrated:
            raise RuntimeError("No pattern-aware components are enabled.")

        stacked = np.stack([calibrated[name] for name in calibrated.keys()], axis=-1)
        if self.aggregation == "mean":
            return stacked.mean(axis=-1)
        if self.aggregation == "max":
            return stacked.max(axis=-1)
        if self.aggregation == "logsumexp":
            tau = max(self.logsumexp_tau, self.eps)
            max_value = np.max(stacked / tau, axis=-1, keepdims=True)
            return tau * (
                np.log(np.exp(stacked / tau - max_value).mean(axis=-1) + self.eps)
                + np.squeeze(max_value, axis=-1)
            )

        k = min(self.top_k, stacked.shape[-1])
        top_values = np.partition(stacked, -k, axis=-1)[..., -k:]
        return top_values.mean(axis=-1)

    def _reliability_weighted_score(self, true_windows, pred_windows):
        if not self.fitted:
            raise RuntimeError("PatternAwareScorer must be fitted before scoring.")

        arrays = self._evidence_arrays(true_windows, pred_windows)
        raw = arrays["raw"]
        if not self.use_calibration:
            return raw

        dynamics = []
        for name in ("scale_context", "trend_context", "freq_context"):
            dynamics.append(np.maximum(self._z(name, arrays[name]), 0.0))
        dynamic_context = np.mean(np.stack(dynamics, axis=-1), axis=-1)

        sync_risk = np.maximum(self._z("sync_context", arrays["sync_context"]), 0.0)
        relief = self.context_strength * np.tanh(dynamic_context)
        risk = self.risk_strength * np.tanh(sync_risk)
        weight = np.clip(1.0 - relief + risk, self.min_weight, self.max_weight)
        return raw * weight

    def score_windows(
        self,
        true_windows,
        pred_windows,
        score_mask=None,
        distribution_params=None,
    ):
        mode = self.score_mode.lower()
        if mode == "auto":
            mode = "tail_probability" if self.distribution != "mse" else "raw"
        if mode in {"nll", "distribution_nll", "conditional_nll"}:
            if self.distribution == "mse":
                return self._raw_residual_score(
                    true_windows, pred_windows, score_mask=score_mask
                )
            cell_nll = self._distribution_nll(
                true_windows, pred_windows, distribution_params
            )
            return self._aggregate_cell_scores(cell_nll, score_mask=score_mask)

        if mode in {
            "tail",
            "tail_probability",
            "tail_surprisal",
            "conditional_tail",
        }:
            if self.distribution == "mse":
                return self._raw_residual_score(
                    true_windows, pred_windows, score_mask=score_mask
                )
            cell_tail = self._distribution_tail_surprisal(
                true_windows, pred_windows, distribution_params
            )
            return self._aggregate_cell_scores(cell_tail, score_mask=score_mask)

        if mode == "raw":
            return self._raw_residual_score(
                true_windows, pred_windows, score_mask=score_mask
            )

        raw_only = self.components == ("raw",) and not self.use_calibration
        if raw_only:
            return self._raw_residual_score(
                true_windows, pred_windows, score_mask=score_mask
            )

        if score_mask is not None:
            # Masked scoring is intentionally supported only for raw residual.
            # Legacy component aggregation remains available as an explicit
            # unmasked ablation path.
            return self._evidence_arrays(true_windows, pred_windows, score_mask=score_mask)["raw"]

        if mode in {"aggregate", "legacy", "component_aggregate"}:
            return self._aggregate_components(true_windows, pred_windows)
        if mode in {"reliability_weighted", "context_weighted"}:
            return self._reliability_weighted_score(true_windows, pred_windows)
        raise ValueError(f"Unknown pattern score_mode: {self.score_mode}")

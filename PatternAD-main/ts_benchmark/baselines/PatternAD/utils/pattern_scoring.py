import numpy as np


class PatternAwareScorer:
    """Pattern-aware post-hoc anomaly scorer for reconstruction residuals.

    The scorer is intentionally model-agnostic: it consumes true windows and
    reconstructed/predicted windows, builds several residual-context evidence
    scores, calibrates them on normal training windows, and produces a final
    point-wise anomaly score per window.
    """

    DEFAULT_COMPONENTS = ("raw", "scale", "trend", "shift", "freq", "sync")

    def __init__(
        self,
        components=None,
        local_window=5,
        trend_window=7,
        aggregation="topk",
        top_k=2,
        logsumexp_tau=1.0,
        eps=1e-6,
        use_calibration=True,
    ):
        self.components = tuple(components or self.DEFAULT_COMPONENTS)
        self.local_window = max(3, int(local_window))
        self.trend_window = max(3, int(trend_window))
        self.aggregation = aggregation
        self.top_k = max(1, int(top_k))
        self.logsumexp_tau = float(logsumexp_tau)
        self.eps = float(eps)
        self.use_calibration = bool(use_calibration)
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

    def component_scores(self, true, pred):
        true = self._as_numpy(true)
        pred = self._as_numpy(pred)
        if true.shape != pred.shape:
            raise ValueError(
                "PatternAwareScorer true/pred shape mismatch: "
                f"{true.shape} vs {pred.shape}"
            )

        residual = (true - pred) ** 2
        raw = residual.mean(axis=-1)

        local_scale = self._rolling_std(true, self.local_window)
        scale = (residual / (local_scale ** 2 + self.eps)).mean(axis=-1)

        true_trend = self._rolling_mean(true, self.trend_window)
        pred_trend = self._rolling_mean(pred, self.trend_window)
        trend = ((true_trend - pred_trend) ** 2).mean(axis=-1)

        true_shift = self._left_right_shift(true, self.trend_window)
        pred_shift = self._left_right_shift(pred, self.trend_window)
        shift = ((true_shift - pred_shift) ** 2).mean(axis=-1)

        true_high = true - true_trend
        pred_high = pred - pred_trend
        freq = ((true_high - pred_high) ** 2).mean(axis=-1)

        median = np.median(residual, axis=-1, keepdims=True)
        mad = np.median(np.abs(residual - median), axis=-1, keepdims=True)
        active = residual > (median + 1.4826 * mad + self.eps)
        active_ratio = active.mean(axis=-1)
        sync = raw * (1.0 + active_ratio)

        all_scores = {
            "raw": raw,
            "scale": scale,
            "trend": trend,
            "shift": shift,
            "freq": freq,
            "sync": sync,
        }
        return {name: all_scores[name] for name in self.components if name in all_scores}

    def fit(self, true_windows, pred_windows):
        scores = self.component_scores(true_windows, pred_windows)
        self.stats = {}
        if not self.use_calibration:
            self.fitted = True
            return self
        for name, value in scores.items():
            flat = np.asarray(value, dtype=np.float64).reshape(-1)
            center = float(np.median(flat))
            scale = float(np.median(np.abs(flat - center)) * 1.4826)
            if scale < self.eps:
                scale = float(np.std(flat))
            if scale < self.eps:
                scale = 1.0
            self.stats[name] = {"center": center, "scale": scale}
        self.fitted = True
        return self

    def transform_components(self, true_windows, pred_windows):
        if not self.fitted:
            raise RuntimeError("PatternAwareScorer must be fitted before scoring.")
        scores = self.component_scores(true_windows, pred_windows)
        if not self.use_calibration:
            return scores
        calibrated = {}
        for name, value in scores.items():
            stat = self.stats[name]
            z = (value - stat["center"]) / (stat["scale"] + self.eps)
            calibrated[name] = np.maximum(z, 0.0)
        return calibrated

    def score_windows(self, true_windows, pred_windows):
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

        # Default: top-k aggregation. This avoids diluting a strong localized
        # evidence source with unrelated weak components.
        k = min(self.top_k, stacked.shape[-1])
        top_values = np.partition(stacked, -k, axis=-1)[..., -k:]
        return top_values.mean(axis=-1)

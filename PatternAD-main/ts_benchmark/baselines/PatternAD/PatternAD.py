import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import optim

from ts_benchmark.baselines.PatternAD.utils.pattern_scoring import PatternAwareScorer
from ts_benchmark.baselines.PatternAD.utils.tools import EarlyStopping, adjust_learning_rate
from ts_benchmark.baselines.utils import anomaly_detection_multi_data_provider, train_val_split


DEFAULT_PATTERN_AD_HYPER_PARAMS = {
    "enc_in": 4,
    "e_layers": 1,
    "d_model": 128,
    "d_ff": 256,
    "lradj": "type1",
    "n_heads": 8,
    "seq_len": 72,
    "win_size": 72,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_size": 16,
    "lr": 0.0001,
    "num_epochs": 3,
    "num_workers": 0,
    "patience": 3,
    "task_name": "anomaly_detection",
    "anomaly_ratio": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    "parallel_strategy": "DP",
    "enc_in_time": 4,
    "backbone_name": "ContextConditionedReconstructor",
    "train_mask_ratio": 0.25,
    "train_variable_mask_ratio": 0.15,
    "reconstruction_full_loss_weight": 0.1,
    "pattern_score_components": ["raw"],
    "pattern_score_local_window": 5,
    "pattern_score_trend_window": 7,
    "pattern_score_aggregation": "mean",
    "pattern_score_top_k": 2,
    "pattern_score_logsumexp_tau": 1.0,
    "pattern_score_eps": 1e-6,
    "pattern_score_use_calibration": False,
    "pattern_score_mode": "raw",
    "pattern_score_context_strength": 0.35,
    "pattern_score_risk_strength": 0.15,
    "pattern_score_min_weight": 0.5,
    "pattern_score_max_weight": 1.5,
    "pattern_score_max_fit_windows": 20000,
    "use_context_conditioning": True,
    "context_window": 7,
    "context_film_strength": 0.2,
    "use_conditional_scoring": True,
    "score_mask_ratio": 0.35,
    "reconstruction_distribution": "mse",
    "distribution_min_scale": 1e-3,
    "distribution_max_scale": 100.0,
    "distribution_init_scale": 1.0,
    "student_t_df": 4.0,
    "student_t_learn_df": True,
    "student_t_max_df": 100.0,
    "train_mask_seed": None,
}


class PatternADConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_PATTERN_AD_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.seq_len = int(getattr(self, "seq_len", getattr(self, "win_size", 72)))
        self.win_size = self.seq_len
        if "train_mask_ratio" not in kwargs and "mask_ratio" in kwargs:
            self.train_mask_ratio = float(kwargs["mask_ratio"])
        else:
            self.train_mask_ratio = float(getattr(self, "train_mask_ratio", getattr(self, "mask_ratio", 0.25)))
        self.train_variable_mask_ratio = float(getattr(self, "train_variable_mask_ratio", 0.15))
        self.reconstruction_full_loss_weight = float(getattr(self, "reconstruction_full_loss_weight", 0.1))
        self.use_context_conditioning = bool(getattr(self, "use_context_conditioning", True))
        self.context_window = max(3, int(getattr(self, "context_window", 7)))
        if self.context_window % 2 == 0:
            self.context_window += 1
        self.context_film_strength = float(getattr(self, "context_film_strength", 0.2))
        self.use_conditional_scoring = bool(getattr(self, "use_conditional_scoring", True))
        self.score_mask_ratio = float(getattr(self, "score_mask_ratio", 0.35))
        if "reconstruction_distribution" in kwargs:
            distribution = kwargs["reconstruction_distribution"]
        elif "distribution_mode" in kwargs:
            distribution = kwargs["distribution_mode"]
        elif "distribution_head" in kwargs:
            distribution = kwargs["distribution_head"]
        else:
            distribution = "mse"
        distribution = str(distribution).lower().replace("-", "_")
        distribution_aliases = {
            "raw": "mse",
            "normal": "gaussian",
            "student": "student_t",
            "studentt": "student_t",
        }
        self.reconstruction_distribution = distribution_aliases.get(distribution, distribution)
        if self.reconstruction_distribution not in {"mse", "gaussian", "student_t"}:
            raise ValueError(
                "reconstruction_distribution must be one of: mse, gaussian, student_t."
            )
        if "pattern_score_mode" not in kwargs and self.reconstruction_distribution != "mse":
            self.pattern_score_mode = "nll"
        self.distribution_min_scale = max(
            float(getattr(self, "distribution_min_scale", 1e-3)), 1e-8
        )
        self.distribution_max_scale = max(
            float(getattr(self, "distribution_max_scale", 100.0)),
            self.distribution_min_scale,
        )
        self.distribution_init_scale = min(
            max(
                float(getattr(self, "distribution_init_scale", 1.0)),
                self.distribution_min_scale,
            ),
            self.distribution_max_scale,
        )
        self.student_t_df = max(float(getattr(self, "student_t_df", 4.0)), 2.01)
        self.student_t_learn_df = bool(getattr(self, "student_t_learn_df", True))
        self.student_t_max_df = max(
            float(getattr(self, "student_t_max_df", 100.0)), self.student_t_df
        )
        configured_mask_seed = getattr(self, "train_mask_seed", None)
        self.train_mask_seed = int(
            torch.initial_seed()
            if configured_mask_seed is None
            else configured_mask_seed
        )
        if self.parallel_strategy not in [None, "DP"]:
            raise ValueError("Invalid value for parallel_strategy. Supported values are 'DP' and None.")

    @property
    def pred_len(self):
        return 0

    @property
    def learning_rate(self):
        return self.lr

    @property
    def model_name(self):
        return "PatternAD"


def _activation_module(name):
    if str(name).lower() == "relu":
        return nn.ReLU()
    return nn.GELU()


class JointMultivariateReconstructor(nn.Module):
    """Context-conditioned denoising reconstructor for multivariate TSAD.

    The model does not use temporal context as a post-hoc score multiplier.
    It encodes local scale, trend, high-frequency residual, and mask structure
    as reconstruction conditions, then reconstructs masked variables from the
    remaining temporal and cross-variable evidence.
    """

    def __init__(self, config):
        super().__init__()
        self.seq_len = int(config.seq_len)
        self.enc_in = int(config.enc_in)
        self.use_context_conditioning = bool(getattr(config, "use_context_conditioning", True))
        self.context_window = max(3, int(getattr(config, "context_window", 7)))
        if self.context_window % 2 == 0:
            self.context_window += 1
        self.context_film_strength = float(getattr(config, "context_film_strength", 0.2))
        self.reconstruction_distribution = str(
            getattr(config, "reconstruction_distribution", "mse")
        )
        self.distribution_min_scale = float(getattr(config, "distribution_min_scale", 1e-3))
        self.distribution_max_scale = float(getattr(config, "distribution_max_scale", 100.0))
        self.student_t_df = float(getattr(config, "student_t_df", 4.0))
        self.student_t_learn_df = bool(getattr(config, "student_t_learn_df", True))
        self.student_t_max_df = float(getattr(config, "student_t_max_df", 100.0))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.enc_in))
        self.input_proj = nn.Linear(self.enc_in, config.d_model)
        self.mask_proj = nn.Linear(self.enc_in, config.d_model)
        context_dim = self.enc_in * 6
        self.context_control = nn.Parameter(torch.zeros(1, 1, context_dim))
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, config.d_ff),
            _activation_module(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.film = nn.Linear(config.d_model, 2 * config.d_model)
        self.pre_encoder_norm = nn.LayerNorm(config.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.e_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            _activation_module(config.activation),
            nn.Dropout(config.dropout),
            # All variants use the same sized head. MSE ignores scale/df, so
            # distribution ablations do not change the parameter count.
            nn.Linear(config.d_ff, 3 * self.enc_in),
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self._init_distribution_bias(config)

    def _init_distribution_bias(self, config):
        output_layer = self.output_proj[-1]
        init_scale = float(getattr(config, "distribution_init_scale", 1.0))
        scale_offset = max(init_scale - self.distribution_min_scale, 1e-6)
        scale_bias = scale_offset if scale_offset > 20.0 else math.log(math.expm1(scale_offset))
        df_offset = max(self.student_t_df - 2.0, 1e-6)
        df_bias = df_offset if df_offset > 20.0 else math.log(math.expm1(df_offset))
        with torch.no_grad():
            output_layer.weight[self.enc_in:, :].zero_()
            output_layer.bias[self.enc_in:2 * self.enc_in].fill_(scale_bias)
            output_layer.bias[2 * self.enc_in:].fill_(df_bias)

    @staticmethod
    def _odd_window(window):
        window = max(3, int(window))
        return window + 1 if window % 2 == 0 else window

    def _masked_rolling_mean(self, x, valid, window):
        window = self._odd_window(window)
        radius = window // 2
        valid = valid.to(dtype=x.dtype)
        values = torch.where(valid.bool(), x, torch.zeros_like(x)).transpose(1, 2)
        weights = valid.transpose(1, 2)
        value_sum = F.avg_pool1d(
            F.pad(values, (radius, radius), mode="replicate"),
            kernel_size=window,
            stride=1,
        ) * window
        weight_sum = F.avg_pool1d(
            F.pad(weights, (radius, radius), mode="replicate"),
            kernel_size=window,
            stride=1,
        ) * window
        mean = value_sum / weight_sum.clamp_min(1.0)
        mean = torch.where(weight_sum > 0, mean, torch.zeros_like(mean))
        return mean.transpose(1, 2)

    def _context_features(self, x, mask):
        valid = torch.ones_like(x, dtype=torch.bool) if mask is None else ~mask.bool()
        local_mean = self._masked_rolling_mean(x, valid, self.context_window)
        local_mean_sq = self._masked_rolling_mean(x * x, valid, self.context_window)
        local_std = torch.sqrt(torch.clamp(local_mean_sq - local_mean * local_mean, min=0.0) + 1e-6)

        diff = torch.zeros_like(x)
        diff[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        diff_valid = torch.zeros_like(valid)
        diff_valid[:, 1:, :] = valid[:, 1:, :] & valid[:, :-1, :]
        trend = self._masked_rolling_mean(diff, diff_valid, self.context_window)
        context_x = torch.where(valid, x, local_mean)
        high_freq = torch.where(valid, x - local_mean, torch.zeros_like(x))
        mask_float = torch.zeros_like(x) if mask is None else mask.float()
        return torch.cat(
            [context_x, local_mean, local_std, trend, high_freq, mask_float], dim=-1
        )

    def _conditioning_features(self, x, mask):
        if self.use_context_conditioning:
            return self._context_features(x, mask)
        return self.context_control.expand(x.shape[0], x.shape[1], -1)

    def forward(self, x, mask=None):
        if x.ndim != 3:
            raise ValueError("JointMultivariateReconstructor expects [B, T, D] input.")
        if x.shape[-1] != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} variables, got {x.shape[-1]}.")
        if x.shape[1] > self.seq_len:
            raise ValueError(f"Expected at most {self.seq_len} time steps, got {x.shape[1]}.")

        if mask is not None:
            mask = mask.bool()
            x_in = torch.where(mask, self.mask_token.expand_as(x), x)
        else:
            x_in = x

        h = self.input_proj(x_in) + self.pos_embedding[:, : x.shape[1], :]
        if mask is not None:
            h = h + self.mask_proj(mask.float())

        # C1 uses target-blind visible context. C0 sends a learned dataset-level
        # constant through the same projection and FiLM path.
        context = self.context_proj(self._conditioning_features(x, mask))
        gamma, beta = self.film(context).chunk(2, dim=-1)
        strength = self.context_film_strength
        h = h + context
        h = h * (1.0 + strength * torch.tanh(gamma)) + strength * beta
        h = self.pre_encoder_norm(h)

        h = self.encoder(h)
        mean, raw_scale, raw_df = self.output_proj(self.norm(h)).chunk(3, dim=-1)
        scale = F.softplus(raw_scale) + self.distribution_min_scale
        scale = scale.clamp(max=self.distribution_max_scale)
        if self.student_t_learn_df:
            df = 2.0 + F.softplus(raw_df)
            df = df.clamp(max=self.student_t_max_df)
        else:
            df = torch.full_like(mean, self.student_t_df)
        if self.reconstruction_distribution == "mse":
            return mean
        return {"mean": mean, "scale": scale, "df": df}


class PatternAD:
    """Multivariate TSAD model with context-conditioned reconstruction."""

    def __init__(self, **kwargs):
        super().__init__()
        self.config = PatternADConfig(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = self.config.seq_len
        self.pattern_scorer = None
        self.model = None
        self.early_stopping = None
        self._train_mask_generators = {}

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        try:
            freq = pd.infer_freq(train_data.index)
        except Exception:
            freq = "S"
        if freq is None:
            raise ValueError("Irregular time intervals")
        self.config.freq = freq[0].lower() if freq[0].lower() in ["m", "w", "b", "d", "h", "t", "s"] else "s"

        column_num = train_data.shape[1]
        if column_num <= 1:
            raise ValueError("PatternAD is designed for multivariate TSAD and requires more than one variable.")
        self.config.enc_in = column_num
        self.config.enc_in_time = column_num

    def _build_pattern_scorer(self):
        return PatternAwareScorer(
            components=getattr(self.config, "pattern_score_components", None),
            local_window=getattr(self.config, "pattern_score_local_window", 5),
            trend_window=getattr(self.config, "pattern_score_trend_window", 7),
            aggregation=getattr(self.config, "pattern_score_aggregation", "mean"),
            top_k=getattr(self.config, "pattern_score_top_k", 2),
            logsumexp_tau=getattr(self.config, "pattern_score_logsumexp_tau", 1.0),
            eps=getattr(self.config, "pattern_score_eps", 1e-6),
            use_calibration=getattr(self.config, "pattern_score_use_calibration", False),
            score_mode=getattr(self.config, "pattern_score_mode", "raw"),
            context_strength=getattr(self.config, "pattern_score_context_strength", 0.35),
            risk_strength=getattr(self.config, "pattern_score_risk_strength", 0.15),
            min_weight=getattr(self.config, "pattern_score_min_weight", 0.5),
            max_weight=getattr(self.config, "pattern_score_max_weight", 1.5),
            distribution=getattr(self.config, "reconstruction_distribution", "mse"),
        )

    @staticmethod
    def _align_output_to_true(outputs, true):
        if isinstance(outputs, dict):
            return {
                name: PatternAD._align_output_to_true(value, true)
                for name, value in outputs.items()
            }
        if outputs.shape == true.shape:
            return outputs
        if outputs.ndim == true.ndim and outputs.shape[:2] == true.shape[:2]:
            return outputs[:, :, -true.shape[-1]:]
        return outputs

    def _forward_backbone(self, batch_x_time, mask=None):
        return self.model(batch_x_time, mask)

    @staticmethod
    def _output_mean(outputs):
        if isinstance(outputs, dict):
            if "mean" not in outputs:
                raise ValueError("Distribution output is missing its mean tensor.")
            return outputs["mean"]
        return outputs

    @staticmethod
    def _distribution_params(outputs):
        if not isinstance(outputs, dict):
            return None
        return {name: outputs[name] for name in ("scale", "df") if name in outputs}

    def _train_mask_generator(self, device):
        key = str(device)
        if key not in self._train_mask_generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.config.train_mask_seed)
            self._train_mask_generators[key] = generator
        return self._train_mask_generators[key]

    def _mask_input(self, batch_x_time):
        ratio = float(getattr(self.config, "train_mask_ratio", 0.0))
        variable_ratio = float(getattr(self.config, "train_variable_mask_ratio", 0.0))
        generator = self._train_mask_generator(batch_x_time.device)
        keep = torch.ones_like(batch_x_time, dtype=torch.bool)
        if ratio > 0:
            point_random = torch.rand(
                batch_x_time.shape,
                dtype=batch_x_time.dtype,
                device=batch_x_time.device,
                generator=generator,
            )
            keep = keep & (point_random > ratio)
        if variable_ratio > 0:
            variable_keep = torch.rand(
                batch_x_time.shape[0],
                1,
                batch_x_time.shape[-1],
                device=batch_x_time.device,
                generator=generator,
            ) > variable_ratio
            keep = keep & variable_keep
        masked_positions = ~keep
        return batch_x_time * keep.float(), masked_positions

    def _complementary_score_masks(self, batch_x_time):
        """Partition every [time, variable] position across score passes."""
        if batch_x_time.shape[-1] <= 1:
            raise ValueError("Complementary scoring requires at least two variables.")
        ratio = float(getattr(self.config, "score_mask_ratio", 0.35))
        ratio = min(max(ratio, 1e-6), 0.8)
        group_count = min(
            batch_x_time.shape[-1],
            max(2, int(math.ceil(1.0 / ratio))),
        )
        t = torch.arange(batch_x_time.shape[1], device=batch_x_time.device).unsqueeze(1)
        d = torch.arange(batch_x_time.shape[2], device=batch_x_time.device).unsqueeze(0)
        assignment = (t + d) % group_count
        return tuple(
            (assignment == group).unsqueeze(0).expand(batch_x_time.shape[0], -1, -1)
            for group in range(group_count)
        )

    def _score_mask_input(self, batch_x_time):
        """Return the first score mask for backward-compatible diagnostics."""
        if not bool(getattr(self.config, "use_conditional_scoring", True)):
            return batch_x_time, None
        score_mask = self._complementary_score_masks(batch_x_time)[0]
        return batch_x_time.masked_fill(score_mask, 0.0), score_mask

    def _predict_for_scoring(self, batch_x_time, force_conditional=False):
        use_conditional = bool(
            force_conditional
            or getattr(self.config, "use_conditional_scoring", True)
        )
        if not use_conditional:
            outputs = self._forward_backbone(batch_x_time, None)
            return self._align_output_to_true(outputs, batch_x_time), None

        combined = None
        coverage = torch.zeros_like(batch_x_time, dtype=torch.int8)
        for score_mask in self._complementary_score_masks(batch_x_time):
            score_input = batch_x_time.masked_fill(score_mask, 0.0)
            current = self._align_output_to_true(
                self._forward_backbone(score_input, score_mask), batch_x_time
            )
            if not isinstance(current, dict):
                current = {"mean": current}
            if combined is None:
                combined = {name: torch.zeros_like(value) for name, value in current.items()}
            mask_float = score_mask.to(dtype=batch_x_time.dtype)
            for name, value in current.items():
                combined[name] = combined[name] + value * mask_float
            coverage = coverage + score_mask.to(dtype=coverage.dtype)

        if not torch.all(coverage == 1):
            raise RuntimeError("Complementary score masks must cover every position exactly once.")
        return combined, coverage.bool()

    def _elementwise_reconstruction_loss(self, outputs, target):
        mean = self._output_mean(outputs)
        distribution = getattr(self.config, "reconstruction_distribution", "mse")
        if distribution == "mse":
            return (mean - target) ** 2
        if not isinstance(outputs, dict) or "scale" not in outputs:
            raise ValueError(f"{distribution} loss requires a predicted scale tensor.")

        scale = outputs["scale"].clamp_min(self.config.distribution_min_scale)
        standardized = (target - mean) / scale
        if distribution == "gaussian":
            return 0.5 * standardized.square() + torch.log(scale) + 0.5 * math.log(2.0 * math.pi)

        if "df" not in outputs:
            raise ValueError("student_t loss requires a predicted df tensor.")
        df = outputs["df"].clamp_min(2.0 + 1e-6)
        return (
            torch.lgamma(0.5 * df)
            - torch.lgamma(0.5 * (df + 1.0))
            + 0.5 * (torch.log(df) + math.log(math.pi))
            + torch.log(scale)
            + 0.5 * (df + 1.0) * torch.log1p(standardized.square() / df)
        )

    def _reconstruction_loss(self, outputs, target, masked_positions):
        elementwise_loss = self._elementwise_reconstruction_loss(outputs, target)
        if masked_positions is not None and masked_positions.any():
            masked_loss = elementwise_loss[masked_positions].mean()
        else:
            masked_loss = elementwise_loss.mean()

        # On visible positions a probabilistic decoder can copy the target and
        # reduce NLL only by collapsing its scale. Keep the auxiliary full-window
        # term on the conditional mean so scale/df are learned from hidden targets.
        mean = self._output_mean(outputs)
        full_mean_loss = ((mean - target) ** 2).mean()
        return (
            masked_loss
            + self.config.reconstruction_full_loss_weight * full_mean_loss
        )

    @staticmethod
    def _windows_to_point_scores(window_scores: np.ndarray, total_len: int) -> np.ndarray:
        window_scores = np.asarray(window_scores)
        if window_scores.ndim == 1:
            return window_scores[:total_len]

        point_sum = np.zeros(total_len, dtype=np.float64)
        point_count = np.zeros(total_len, dtype=np.float64)
        for start, scores in enumerate(window_scores):
            if start >= total_len:
                break
            end = min(start + len(scores), total_len)
            point_sum[start:end] += scores[: end - start]
            point_count[start:end] += 1

        point_scores = np.zeros(total_len, dtype=np.float64)
        valid = point_count > 0
        point_scores[valid] = point_sum[valid] / point_count[valid]
        if not valid.all() and valid.any():
            point_scores[~valid] = point_scores[valid][-1]
        return point_scores

    def _collect_multi_scores(self, data_loader, total_len: int) -> np.ndarray:
        if self.pattern_scorer is None or not self.pattern_scorer.fitted:
            raise RuntimeError("Pattern scorer is not fitted. Call detect_multi_fit before scoring.")

        window_scores = []
        with torch.no_grad():
            for batch_x_time, _, _, _ in data_loader:
                batch_x_time = batch_x_time.float().to(self.device)
                outputs, score_mask = self._predict_for_scoring(batch_x_time)
                window_scores.append(
                    self.pattern_scorer.score_windows(
                        batch_x_time,
                        self._output_mean(outputs),
                        score_mask=score_mask,
                        distribution_params=self._distribution_params(outputs),
                    )
                )

        return self._windows_to_point_scores(np.concatenate(window_scores, axis=0), total_len)

    def _fit_pattern_scorer(self, data_loader):
        self.pattern_scorer = self._build_pattern_scorer()
        score_mode = str(getattr(self.config, "pattern_score_mode", "raw")).lower()
        if score_mode in {
            "raw",
            "nll",
            "distribution_nll",
            "conditional_nll",
            "auto",
        }:
            # Raw residual and distribution NLL do not use fitted legacy
            # component statistics. Avoid materializing large float64 window
            # tensors solely to toggle the scorer's fitted guard.
            self.pattern_scorer.fitted = True
            return

        self.model.load_state_dict(self.early_stopping.check_point)
        self.model.to(self.device)
        self.model.eval()

        true_windows, pred_windows = [], []
        max_windows = int(getattr(self.config, "pattern_score_max_fit_windows", 20000))
        collected_windows = 0
        with torch.no_grad():
            for batch_x_time, _, _, _ in data_loader:
                batch_x_time = batch_x_time.float().to(self.device)
                outputs, _ = self._predict_for_scoring(batch_x_time)

                take = min(batch_x_time.shape[0], max_windows - collected_windows)
                if take <= 0:
                    break
                true_windows.append(batch_x_time[:take].detach().cpu().numpy())
                pred_windows.append(self._output_mean(outputs)[:take].detach().cpu().numpy())
                collected_windows += take
                if collected_windows >= max_windows:
                    break

        if not true_windows:
            raise RuntimeError("No windows collected for pattern-aware scorer calibration.")

        self.pattern_scorer.fit(
            np.concatenate(true_windows, axis=0),
            np.concatenate(pred_windows, axis=0),
        )
        self.model.train()

    def detect_multi_validate(self, valid_data_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x_time, _, _, _ in valid_data_loader:
                batch_x_time = batch_x_time.float().to(self.device)
                # Checkpoint selection is held fixed across A/B cells. The M
                # factor changes final scoring only, not validation semantics.
                outputs, valid_mask = self._predict_for_scoring(
                    batch_x_time, force_conditional=True
                )
                loss = self._reconstruction_loss(outputs, batch_x_time, valid_mask)
                total_loss.append(loss.detach().cpu().numpy())

        self.model.train()
        return np.mean(total_loss)

    def detect_multi_fit(self, train_data: pd.DataFrame, train_text: pd.DataFrame, train_label: pd.DataFrame):
        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.model = JointMultivariateReconstructor(self.config)
        self._train_mask_generators = {}

        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        train_data_text, valid_text = train_val_split(train_text, 0.8, None)
        self.scaler.fit(train_data_value.values)

        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )
        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )
        train_data_text = pd.DataFrame(train_data_text, columns=train_data_text.columns, index=train_data_text.index)
        valid_text = pd.DataFrame(valid_text, columns=valid_text.columns, index=valid_text.index)
        self.train_data_value = train_data_value
        self.train_data_text = train_data_text

        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(self.config.train_mask_seed)
        valid_loader_generator = torch.Generator()
        valid_loader_generator.manual_seed(self.config.train_mask_seed + 1)
        self.valid_data_loader = anomaly_detection_multi_data_provider(
            valid_data,
            valid_text,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="val",
            generator=valid_loader_generator,
        )
        self.train_data_loader = anomaly_detection_multi_data_provider(
            train_data_value,
            train_data_text,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="train",
            generator=train_loader_generator,
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.early_stopping = EarlyStopping(patience=self.config.patience)
        self.model.to(self.device)

        time_now = time.time()
        for epoch in range(self.config.num_epochs):
            iter_count = 0
            self.model.train()
            for i, (batch_x_time, _, _, _) in enumerate(self.train_data_loader):
                iter_count += 1
                train_steps = len(self.train_data_loader)
                optimizer.zero_grad()
                batch_x_time = batch_x_time.float().to(self.device)
                masked_x_time, masked_positions = self._mask_input(batch_x_time)
                outputs = self._forward_backbone(masked_x_time, masked_positions)
                outputs = self._align_output_to_true(outputs, batch_x_time)
                loss = self._reconstruction_loss(outputs, batch_x_time, masked_positions)

                if (i + 1) % 10 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.config.num_epochs - epoch) * train_steps - i)
                    print(f"\titers: {i + 1}, epoch: {epoch + 1}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                optimizer.step()

            valid_loss = self.detect_multi_validate(self.valid_data_loader, criterion)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break
            adjust_learning_rate(optimizer, epoch + 1, self.config)

        pattern_fit_loader = anomaly_detection_multi_data_provider(
            train_data_value,
            train_data_text,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="test",
        )
        self._fit_pattern_scorer(pattern_fit_loader)

    def detect_multi_score(self, test_data: pd.DataFrame, test_text: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call detect_multi_fit first.")
        self.model.load_state_dict(self.early_stopping.check_point)

        test_data = pd.DataFrame(
            self.scaler.transform(test_data.values), columns=test_data.columns, index=test_data.index
        )
        test_text = pd.DataFrame(test_text.values, columns=test_text.columns, index=test_text.index)
        test_data_loader = anomaly_detection_multi_data_provider(
            test_data,
            test_text,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="test",
        )

        self.model.to(self.device)
        self.model.eval()
        test_energy = self._collect_multi_scores(test_data_loader, len(test_data))
        return test_energy, test_energy

    def detect_multi_label(self, test_data: pd.DataFrame, test_text: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call detect_multi_fit first.")
        self.model.load_state_dict(self.early_stopping.check_point)

        test_data = pd.DataFrame(
            self.scaler.transform(test_data.values), columns=test_data.columns, index=test_data.index
        )
        test_text = pd.DataFrame(test_text.values, columns=test_text.columns, index=test_text.index)
        test_data_loader = anomaly_detection_multi_data_provider(
            test_data,
            test_text,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="test",
        )
        train_score_loader = anomaly_detection_multi_data_provider(
            self.train_data_value,
            self.train_data_text,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="test",
        )

        self.model.to(self.device)
        self.model.eval()
        train_energy = self._collect_multi_scores(train_score_loader, len(self.train_data_value))
        test_energy = self._collect_multi_scores(test_data_loader, len(test_data))
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)
        return preds, test_energy

    def __repr__(self) -> str:
        return "PatternAD"

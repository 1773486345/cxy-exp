import copy
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
    "reconstruction_transition_loss_weight": 0.0,
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
    "use_context_scale_prior": True,
    "context_scale_floor": 0.05,
    "context_scale_prior_mix": 0.5,
    "context_scale_log_residual_limit": 1.5,
    "context_transition_scale_suppression": 0.0,
    "use_context_scale_normalization": True,
    "use_context_transition_scale_prior": True,
    "context_transition_scale_floor": 0.05,
    "context_transition_scale_prior_mix": 0.5,
    "context_transition_scale_log_residual_limit": 1.5,
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
        self.reconstruction_transition_loss_weight = max(
            float(getattr(self, "reconstruction_transition_loss_weight", 0.0)),
            0.0,
        )
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
            self.pattern_score_mode = "tail_probability"
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
        self.use_context_scale_prior = bool(
            getattr(self, "use_context_scale_prior", True)
        )
        self.context_scale_floor = max(
            float(getattr(self, "context_scale_floor", 0.05)),
            self.distribution_min_scale,
        )
        self.context_scale_prior_mix = min(
            max(float(getattr(self, "context_scale_prior_mix", 0.5)), 0.0),
            1.0,
        )
        self.context_scale_log_residual_limit = max(
            float(getattr(self, "context_scale_log_residual_limit", 1.5)), 0.0
        )
        self.context_transition_scale_suppression = max(
            float(getattr(self, "context_transition_scale_suppression", 0.0)),
            0.0,
        )
        self.use_context_scale_normalization = bool(
            getattr(self, "use_context_scale_normalization", True)
        )
        self.use_context_transition_scale_prior = bool(
            getattr(self, "use_context_transition_scale_prior", True)
        )
        self.context_transition_scale_floor = max(
            float(getattr(self, "context_transition_scale_floor", 0.05)),
            self.distribution_min_scale,
        )
        self.context_transition_scale_prior_mix = min(
            max(
                float(getattr(self, "context_transition_scale_prior_mix", 0.5)),
                0.0,
            ),
            1.0,
        )
        self.context_transition_scale_log_residual_limit = max(
            float(
                getattr(
                    self, "context_transition_scale_log_residual_limit", 1.5
                )
            ),
            0.0,
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
        self.use_context_scale_prior = bool(
            getattr(config, "use_context_scale_prior", True)
        )
        self.context_scale_floor = float(getattr(config, "context_scale_floor", 0.05))
        self.context_scale_prior_mix = float(
            getattr(config, "context_scale_prior_mix", 0.5)
        )
        self.context_scale_log_residual_limit = float(
            getattr(config, "context_scale_log_residual_limit", 1.5)
        )
        self.context_transition_scale_suppression = float(
            getattr(config, "context_transition_scale_suppression", 0.0)
        )
        self.use_context_scale_normalization = bool(
            getattr(config, "use_context_scale_normalization", True)
        )
        self.use_context_transition_scale_prior = bool(
            getattr(config, "use_context_transition_scale_prior", True)
        )
        self.context_transition_scale_floor = float(
            getattr(config, "context_transition_scale_floor", 0.05)
        )
        self.context_transition_scale_prior_mix = float(
            getattr(config, "context_transition_scale_prior_mix", 0.5)
        )
        self.context_transition_scale_log_residual_limit = float(
            getattr(config, "context_transition_scale_log_residual_limit", 1.5)
        )
        self.student_t_df = float(getattr(config, "student_t_df", 4.0))
        self.student_t_learn_df = bool(getattr(config, "student_t_learn_df", True))
        self.student_t_max_df = float(getattr(config, "student_t_max_df", 100.0))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.enc_in))
        self.input_proj = nn.Linear(self.enc_in, config.d_model)
        self.mask_proj = nn.Linear(self.enc_in, config.d_model)
        context_dim = self.enc_in * 6
        self.context_control = nn.Parameter(torch.zeros(1, 1, context_dim))
        self.scale_control = nn.Parameter(torch.zeros(1, 1, self.enc_in))
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
        self.transition_output_proj = nn.Linear(config.d_model, 2 * self.enc_in)
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
            self.scale_control.fill_(math.log(max(init_scale, 1e-8)))
            if self.use_context_scale_prior:
                # The scale head learns a bounded log correction around the
                # visible-context prior. Zero therefore means no correction.
                output_layer.bias[self.enc_in:2 * self.enc_in].zero_()
            else:
                output_layer.bias[self.enc_in:2 * self.enc_in].fill_(scale_bias)
            output_layer.bias[2 * self.enc_in:].fill_(df_bias)
            self.transition_output_proj.weight.zero_()
            self.transition_output_proj.bias[: self.enc_in].zero_()
            if self.use_context_transition_scale_prior:
                self.transition_output_proj.bias[self.enc_in:].zero_()
            else:
                transition_scale = max(init_scale * math.sqrt(2.0), 1e-6)
                transition_bias = (
                    transition_scale
                    if transition_scale > 20.0
                    else math.log(math.expm1(transition_scale))
                )
                self.transition_output_proj.bias[self.enc_in:].fill_(
                    transition_bias
                )

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

    def _visible_context_statistics(self, x, mask):
        valid = torch.ones_like(x, dtype=torch.bool) if mask is None else ~mask.bool()
        local_mean = self._masked_rolling_mean(x, valid, self.context_window)
        local_mean_sq = self._masked_rolling_mean(x * x, valid, self.context_window)
        local_std = torch.sqrt(torch.clamp(local_mean_sq - local_mean * local_mean, min=0.0) + 1e-6)

        radius = max(self.context_window // 2, 1)
        time_index = torch.arange(
            x.shape[1], dtype=x.dtype, device=x.device
        ).reshape(1, -1, 1) / float(radius)
        time_index = time_index.expand_as(x)
        local_time = self._masked_rolling_mean(time_index, valid, self.context_window)
        local_time_sq = self._masked_rolling_mean(
            time_index * time_index, valid, self.context_window
        )
        local_time_value = self._masked_rolling_mean(
            time_index * x, valid, self.context_window
        )
        time_variance = (local_time_sq - local_time * local_time).clamp_min(1e-6)
        trend = (local_time_value - local_time * local_mean) / time_variance
        transition = torch.zeros_like(x)
        transition_valid = torch.zeros_like(valid)
        transition[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        transition_valid[:, 1:, :] = valid[:, 1:, :] & valid[:, :-1, :]
        local_transition_mean = self._masked_rolling_mean(
            transition, transition_valid, self.context_window
        )
        local_transition_mean_sq = self._masked_rolling_mean(
            transition.square(), transition_valid, self.context_window
        )
        local_transition_std = torch.sqrt(
            (
                local_transition_mean_sq
                - local_transition_mean.square()
            ).clamp_min(0.0)
            + 1e-6
        )
        past_sum = torch.zeros_like(x)
        past_sum_sq = torch.zeros_like(x)
        past_count = torch.zeros_like(x)
        future_sum = torch.zeros_like(x)
        future_sum_sq = torch.zeros_like(x)
        future_count = torch.zeros_like(x)
        for offset in range(1, radius + 1):
            past_values = x[:, :-offset, :]
            past_valid = valid[:, :-offset, :].to(dtype=x.dtype)
            past_sum[:, offset:, :] += past_values * past_valid
            past_sum_sq[:, offset:, :] += past_values.square() * past_valid
            past_count[:, offset:, :] += past_valid

            future_values = x[:, offset:, :]
            future_valid = valid[:, offset:, :].to(dtype=x.dtype)
            future_sum[:, :-offset, :] += future_values * future_valid
            future_sum_sq[:, :-offset, :] += future_values.square() * future_valid
            future_count[:, :-offset, :] += future_valid
        past_mean = past_sum / past_count.clamp_min(1.0)
        future_mean = future_sum / future_count.clamp_min(1.0)
        past_std = torch.sqrt(
            (past_sum_sq / past_count.clamp_min(1.0) - past_mean.square())
            .clamp_min(0.0)
            + 1e-6
        )
        future_std = torch.sqrt(
            (future_sum_sq / future_count.clamp_min(1.0) - future_mean.square())
            .clamp_min(0.0)
            + 1e-6
        )
        change_point_score = (future_mean - past_mean).abs() / (
            torch.sqrt(past_std.square() + future_std.square())
            + self.context_transition_scale_floor
        )
        change_point_score = torch.where(
            (past_count > 0) & (future_count > 0),
            change_point_score,
            torch.zeros_like(change_point_score),
        )
        context_x = torch.where(valid, x, local_mean)
        high_freq = torch.where(valid, x - local_mean, torch.zeros_like(x))
        return {
            "valid": valid,
            "context_x": context_x,
            "local_mean": local_mean,
            "local_std": local_std,
            "trend": trend,
            "local_transition_mean": local_transition_mean,
            "local_transition_std": local_transition_std,
            "change_point_score": change_point_score,
            "high_freq": high_freq,
        }

    def _context_features_from_statistics(self, statistics, mask):
        x = statistics["context_x"]
        mask_float = torch.zeros_like(x) if mask is None else mask.float()
        return torch.cat(
            [
                statistics["context_x"],
                statistics["local_mean"],
                statistics["local_std"],
                statistics["trend"],
                statistics["high_freq"],
                mask_float,
            ],
            dim=-1,
        )

    def _context_features(self, x, mask):
        statistics = self._visible_context_statistics(x, mask)
        return self._context_features_from_statistics(statistics, mask)

    def _conditioning_features(self, x, mask):
        if self.use_context_conditioning:
            return self._context_features(x, mask)
        return self.context_control.expand(x.shape[0], x.shape[1], -1)

    def _contextual_scale_prior(self, x, statistics):
        learned_scale = torch.exp(self.scale_control).expand_as(x)
        if not self.use_context_conditioning or statistics is None:
            return learned_scale

        local_scale = statistics["local_std"].clamp_min(self.context_scale_floor)
        suppression = self.context_transition_scale_suppression
        if suppression > 0.0:
            transition_ratio = statistics["trend"].abs() / (
                local_scale + self.context_scale_floor
            )
            stationary_fraction = torch.exp(-suppression * transition_ratio)
            local_scale = self.context_scale_floor + (
                local_scale - self.context_scale_floor
            ) * stationary_fraction
        mix = self.context_scale_prior_mix
        if mix <= 0.0:
            return learned_scale
        # Geometric interpolation makes the prior multiplicative and keeps
        # its units consistent with the standardized reconstruction target.
        return learned_scale * torch.exp(mix * torch.log(local_scale))

    def _contextual_transition_scale_prior(self, x, statistics):
        learned_scale = (
            math.sqrt(2.0) * torch.exp(self.scale_control).expand_as(x)
        )
        if not self.use_context_conditioning or statistics is None:
            return learned_scale
        local_scale = statistics["local_transition_std"].clamp_min(
            self.context_transition_scale_floor
        )
        mix = self.context_transition_scale_prior_mix
        if mix <= 0.0:
            return learned_scale
        return learned_scale * torch.exp(mix * torch.log(local_scale))

    def _contextual_transition_gate(self, x, statistics):
        if not self.use_context_conditioning or statistics is None:
            return torch.zeros_like(x)
        return torch.tanh(statistics["change_point_score"])

    def forward(self, x, mask=None):
        if x.ndim != 3:
            raise ValueError("JointMultivariateReconstructor expects [B, T, D] input.")
        if x.shape[-1] != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} variables, got {x.shape[-1]}.")
        if x.shape[1] > self.seq_len:
            raise ValueError(f"Expected at most {self.seq_len} time steps, got {x.shape[1]}.")

        statistics = None
        if self.use_context_conditioning:
            statistics = self._visible_context_statistics(x, mask)
        if self.use_context_scale_normalization:
            normalization_scale = self._contextual_scale_prior(x, statistics).clamp(
                min=self.distribution_min_scale,
                max=self.distribution_max_scale,
            )
            normalized_x = x / normalization_scale
        else:
            normalization_scale = torch.ones_like(x)
            normalized_x = x

        if mask is not None:
            mask = mask.bool()
            x_in = torch.where(mask, self.mask_token.expand_as(x), normalized_x)
        else:
            x_in = normalized_x

        h = self.input_proj(x_in) + self.pos_embedding[:, : x.shape[1], :]
        if mask is not None:
            h = h + self.mask_proj(mask.float())

        # C1 uses target-blind visible context. C0 sends a learned dataset-level
        # constant through the same projection and FiLM path.
        if self.use_context_conditioning:
            conditioning = self._context_features_from_statistics(statistics, mask)
        else:
            conditioning = self.context_control.expand(x.shape[0], x.shape[1], -1)
        context = self.context_proj(conditioning)
        gamma, beta = self.film(context).chunk(2, dim=-1)
        strength = self.context_film_strength
        h = h + context
        h = h * (1.0 + strength * torch.tanh(gamma)) + strength * beta
        h = self.pre_encoder_norm(h)

        h = self.encoder(h)
        decoded = self.norm(h)
        mean, raw_scale, raw_df = self.output_proj(decoded).chunk(3, dim=-1)
        transition_mean, raw_transition_scale = self.transition_output_proj(
            decoded
        ).chunk(2, dim=-1)
        if self.use_context_scale_normalization:
            mean = mean * normalization_scale
        if self.use_context_scale_prior:
            scale_prior = self._contextual_scale_prior(x, statistics)
            correction = torch.exp(
                self.context_scale_log_residual_limit * torch.tanh(raw_scale)
            )
            scale = scale_prior * correction
        else:
            scale = F.softplus(raw_scale) + self.distribution_min_scale
        scale = scale.clamp_min(self.distribution_min_scale)
        scale = scale.clamp(max=self.distribution_max_scale)
        if self.use_context_transition_scale_prior:
            transition_scale_prior = self._contextual_transition_scale_prior(
                x, statistics
            )
            transition_mean = transition_mean * transition_scale_prior
            transition_correction = torch.exp(
                self.context_transition_scale_log_residual_limit
                * torch.tanh(raw_transition_scale)
            )
            transition_scale = transition_scale_prior * transition_correction
        else:
            transition_scale = (
                F.softplus(raw_transition_scale) + self.distribution_min_scale
            )
        transition_scale = transition_scale.clamp(
            min=self.distribution_min_scale,
            max=self.distribution_max_scale,
        )
        if self.student_t_learn_df:
            df = 2.0 + F.softplus(raw_df)
            df = df.clamp(max=self.student_t_max_df)
        else:
            df = torch.full_like(mean, self.student_t_df)
        if self.reconstruction_distribution == "mse":
            return mean
        result = {"mean": mean, "scale": scale, "df": df}
        if self.reconstruction_distribution == "gaussian":
            result["transition_mean"] = transition_mean
            result["transition_scale"] = transition_scale
            result["transition_gate"] = self._contextual_transition_gate(
                x, statistics
            )
        return result


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
        self._fit_diagnostics = None
        self._score_call_diagnostics = []
        self._last_score_components = {}

    def get_diagnostics(self):
        return {
            "schema_version": 1,
            "model": "PatternAD",
            "distribution": self.config.reconstruction_distribution,
            "score_mode": str(self.config.pattern_score_mode),
            "training": copy.deepcopy(self._fit_diagnostics),
            "score_calls": copy.deepcopy(self._score_call_diagnostics),
        }

    def get_last_score_components(self):
        return {
            name: np.asarray(values, dtype=np.float64).copy()
            for name, values in self._last_score_components.items()
        }

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
        return {
            name: outputs[name]
            for name in (
                "scale",
                "df",
                "transition_mean",
                "transition_scale",
                "transition_gate",
            )
            if name in outputs
        }

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
        loss = (
            masked_loss
            + self.config.reconstruction_full_loss_weight * full_mean_loss
        )
        transition_weight = self.config.reconstruction_transition_loss_weight
        if transition_weight > 0.0 and target.shape[1] > 1:
            if isinstance(outputs, dict) and "transition_mean" in outputs:
                mean_transition = outputs["transition_mean"][:, 1:, :]
            else:
                mean_transition = mean[:, 1:, :] - mean[:, :-1, :]
            target_transition = target[:, 1:, :] - target[:, :-1, :]
            transition_squared = (mean_transition - target_transition).square()
            if masked_positions is None:
                transition_mask = torch.ones_like(
                    transition_squared, dtype=torch.bool
                )
            else:
                transition_mask = masked_positions[:, 1:, :]
            distribution = self.config.reconstruction_distribution
            if (
                distribution == "gaussian"
                and isinstance(outputs, dict)
                and "transition_scale" in outputs
            ):
                transition_scale = outputs["transition_scale"][:, 1:, :].clamp_min(
                    self.config.distribution_min_scale
                )
                transition_elementwise = (
                    0.5 * transition_squared / transition_scale.square()
                    + torch.log(transition_scale)
                    + 0.5 * math.log(2.0 * math.pi)
                )
            else:
                transition_elementwise = transition_squared
            if transition_mask.any():
                transition_loss = transition_elementwise[transition_mask].mean()
                loss = loss + transition_weight * transition_loss
        return loss

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

        started_at = time.perf_counter()
        window_scores = []
        window_components = {
            "raw_squared_residual": [],
            "standardized_squared_residual": [],
            "predicted_scale": [],
            "log_scale": [],
            "transition_squared_residual": [],
            "transition_standardized_squared_residual": [],
            "predicted_transition_scale": [],
            "transition_gate": [],
        }
        batch_count = 0
        window_count = 0
        scale_count = 0
        scale_finite_count = 0
        scale_sum = 0.0
        scale_sum_sq = 0.0
        scale_min = math.inf
        scale_max = -math.inf
        scale_lower_count = 0
        scale_upper_count = 0
        with torch.no_grad():
            for batch_x_time, _, _, _ in data_loader:
                batch_x_time = batch_x_time.float().to(self.device)
                outputs, score_mask = self._predict_for_scoring(batch_x_time)
                mean_output = self._output_mean(outputs)
                if isinstance(outputs, dict) and "scale" in outputs:
                    component_scale = outputs["scale"].clamp_min(
                        self.config.distribution_min_scale
                    )
                else:
                    component_scale = torch.ones_like(mean_output)
                raw_squared = (batch_x_time - mean_output).square()
                transition_squared = torch.zeros_like(raw_squared)
                if isinstance(outputs, dict) and "transition_scale" in outputs:
                    transition_scale = outputs["transition_scale"].detach().clone()
                else:
                    transition_scale = torch.ones_like(component_scale)
                if batch_x_time.shape[1] > 1:
                    observed_transition = (
                        batch_x_time[:, 1:, :] - batch_x_time[:, :-1, :]
                    )
                    if isinstance(outputs, dict) and "transition_mean" in outputs:
                        predicted_transition = outputs["transition_mean"][:, 1:, :]
                    else:
                        predicted_transition = (
                            mean_output[:, 1:, :] - mean_output[:, :-1, :]
                        )
                    transition_squared[:, 1:, :] = (
                        observed_transition - predicted_transition
                    ).square()
                    if not (
                        isinstance(outputs, dict)
                        and "transition_scale" in outputs
                    ):
                        transition_scale[:, 1:, :] = torch.sqrt(
                            component_scale[:, 1:, :].square()
                            + component_scale[:, :-1, :].square()
                        )
                    transition_squared[:, 0, :] = transition_squared[:, 1, :]
                    transition_scale[:, 0, :] = transition_scale[:, 1, :]
                component_values = {
                    "raw_squared_residual": raw_squared,
                    "standardized_squared_residual": raw_squared
                    / component_scale.square(),
                    "predicted_scale": component_scale,
                    "log_scale": torch.log(component_scale),
                    "transition_squared_residual": transition_squared,
                    "transition_standardized_squared_residual": transition_squared
                    / transition_scale.square(),
                    "predicted_transition_scale": transition_scale,
                    "transition_gate": (
                        outputs["transition_gate"]
                        if isinstance(outputs, dict)
                        and "transition_gate" in outputs
                        else torch.zeros_like(component_scale)
                    ),
                }
                if score_mask is None:
                    aggregated_components = {
                        name: values.mean(dim=-1)
                        for name, values in component_values.items()
                    }
                else:
                    component_mask = score_mask.to(dtype=batch_x_time.dtype)
                    component_count = component_mask.sum(dim=-1).clamp_min(1.0)
                    aggregated_components = {
                        name: (values * component_mask).sum(dim=-1)
                        / component_count
                        for name, values in component_values.items()
                    }
                for name, values in aggregated_components.items():
                    window_components[name].append(values.detach().cpu().numpy())
                batch_count += 1
                window_count += int(batch_x_time.shape[0])
                if isinstance(outputs, dict) and "scale" in outputs:
                    scale = outputs["scale"].detach()
                    finite = torch.isfinite(scale)
                    finite_count = int(finite.sum().item())
                    scale_count += int(scale.numel())
                    scale_finite_count += finite_count
                    if finite_count:
                        finite_scale = scale[finite].double()
                        scale_sum += float(finite_scale.sum().item())
                        scale_sum_sq += float(finite_scale.square().sum().item())
                        scale_min = min(scale_min, float(finite_scale.min().item()))
                        scale_max = max(scale_max, float(finite_scale.max().item()))
                        lower_cutoff = self.config.distribution_min_scale * (1.0 + 1e-6)
                        upper_cutoff = self.config.distribution_max_scale * (1.0 - 1e-7)
                        scale_lower_count += int(
                            (finite_scale <= lower_cutoff).sum().item()
                        )
                        scale_upper_count += int(
                            (finite_scale >= upper_cutoff).sum().item()
                        )
                window_scores.append(
                    self.pattern_scorer.score_windows(
                        batch_x_time,
                        mean_output,
                        score_mask=score_mask,
                        distribution_params=self._distribution_params(outputs),
                    )
                )

        point_scores = self._windows_to_point_scores(
            np.concatenate(window_scores, axis=0), total_len
        )
        self._last_score_components = {
            name: self._windows_to_point_scores(
                np.concatenate(values, axis=0), total_len
            )
            for name, values in window_components.items()
        }
        finite_scores = point_scores[np.isfinite(point_scores)]
        score_diagnostics = {
            "call_index": len(self._score_call_diagnostics),
            "phase": None,
            "input_length": int(total_len),
            "batch_count": int(batch_count),
            "window_count": int(window_count),
            "elapsed_seconds": float(time.perf_counter() - started_at),
            "score": {
                "count": int(point_scores.size),
                "finite_count": int(finite_scores.size),
                "nonfinite_count": int(point_scores.size - finite_scores.size),
                "min": float(finite_scores.min()) if finite_scores.size else None,
                "max": float(finite_scores.max()) if finite_scores.size else None,
                "mean": float(finite_scores.mean()) if finite_scores.size else None,
            },
            "scale": None,
        }
        if scale_count:
            nonfinite_count = scale_count - scale_finite_count
            mean = scale_sum / scale_finite_count if scale_finite_count else None
            variance = (
                max(scale_sum_sq / scale_finite_count - mean * mean, 0.0)
                if scale_finite_count
                else None
            )
            score_diagnostics["scale"] = {
                "count": int(scale_count),
                "finite_count": int(scale_finite_count),
                "nonfinite_count": int(nonfinite_count),
                "min": float(scale_min) if scale_finite_count else None,
                "max": float(scale_max) if scale_finite_count else None,
                "mean": float(mean) if mean is not None else None,
                "std": float(math.sqrt(variance)) if variance is not None else None,
                "lower_bound": float(self.config.distribution_min_scale),
                "upper_bound": float(self.config.distribution_max_scale),
                "lower_bound_count": int(scale_lower_count),
                "upper_bound_count": int(scale_upper_count),
                "lower_bound_fraction": float(scale_lower_count / scale_count),
                "upper_bound_fraction": float(scale_upper_count / scale_count),
            }
        self._score_call_diagnostics.append(score_diagnostics)
        return point_scores

    def _fit_pattern_scorer(self, data_loader):
        self.pattern_scorer = self._build_pattern_scorer()
        score_mode = str(getattr(self.config, "pattern_score_mode", "raw")).lower()
        if score_mode in {
            "raw",
            "nll",
            "distribution_nll",
            "conditional_nll",
            "tail",
            "tail_probability",
            "tail_surprisal",
            "conditional_tail",
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
        fit_started_at = time.perf_counter()
        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.model = JointMultivariateReconstructor(self.config)
        self._train_mask_generators = {}
        self._score_call_diagnostics = []
        self._last_score_components = {}

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
        training_started_at = time.perf_counter()
        epoch_history = []
        best_epoch = None
        for epoch in range(self.config.num_epochs):
            epoch_started_at = time.perf_counter()
            iter_count = 0
            epoch_losses = []
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
                epoch_losses.append(float(loss.detach().cpu().item()))

                if (i + 1) % 10 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.config.num_epochs - epoch) * train_steps - i)
                    print(f"\titers: {i + 1}, epoch: {epoch + 1}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                optimizer.step()

            valid_loss = float(
                self.detect_multi_validate(self.valid_data_loader, criterion)
            )
            previous_best = float(self.early_stopping.val_loss_min)
            self.early_stopping(valid_loss, self.model)
            if float(self.early_stopping.val_loss_min) < previous_best:
                best_epoch = epoch + 1
            epoch_history.append(
                {
                    "epoch": int(epoch + 1),
                    "train_loss": float(np.mean(epoch_losses)),
                    "validation_loss": valid_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_seconds": float(time.perf_counter() - epoch_started_at),
                }
            )
            if self.early_stopping.early_stop:
                break
            adjust_learning_rate(optimizer, epoch + 1, self.config)

        training_seconds = time.perf_counter() - training_started_at
        pattern_fit_loader = anomaly_detection_multi_data_provider(
            train_data_value,
            train_data_text,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="test",
        )
        scorer_started_at = time.perf_counter()
        self._fit_pattern_scorer(pattern_fit_loader)
        scorer_fit_seconds = time.perf_counter() - scorer_started_at
        self._fit_diagnostics = {
            "fit_seconds": float(time.perf_counter() - fit_started_at),
            "training_seconds": float(training_seconds),
            "scorer_fit_seconds": float(scorer_fit_seconds),
            "epochs_requested": int(self.config.num_epochs),
            "epochs_completed": int(len(epoch_history)),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_validation_loss": float(self.early_stopping.val_loss_min),
            "stopped_early": bool(self.early_stopping.early_stop),
            "parameter_count": int(
                sum(parameter.numel() for parameter in self.model.parameters())
            ),
            "optimization_train_points": int(len(train_data_value)),
            "validation_points": int(len(valid_data)),
            "epoch_history": epoch_history,
        }

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

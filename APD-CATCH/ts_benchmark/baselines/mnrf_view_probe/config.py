"""Configuration for the independent frozen-feature view probe experiment."""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig


VALID_VIEWS = ("S", "E", "P", "R")


@dataclass
class ViewProbeConfig:
    seed: int = 2021
    seq_len: int = 128
    pattern_len: int = 32
    hidden_dim: int = 32
    view_dim: int = 8
    evolution_dim: Optional[int] = None
    pattern_dim: Optional[int] = None
    relation_dim: Optional[int] = None
    batch_size: int = 64
    stage1_epochs: int = 50
    stage2_epochs: int = 30
    patience: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    train_noise_std: float = 0.01
    noise_seed: int = 20261013
    feature_std_floor: float = 0.05
    score_mad_floor: float = 0.001
    calibration_quantile: float = 0.995
    cache_chunk_size: int = 256
    encoding_batch_size: int = 0
    device: str = "auto"
    output_dir: str = ""
    resume_from: str = ""
    data_source: str = "large_detect"
    task_name: str = ""
    mode: str = "full_gmm"
    selected_views: Tuple[str, ...] = VALID_VIEWS
    source_dir: str = ""
    gmm_components: int = 4
    gmm_covariance_type: str = "full"
    gmm_reg_covar: float = 0.001
    gmm_n_init: int = 1
    gmm_max_iter: int = 100
    gmm_fit_max_windows: int = 20000
    cpu_threads: int = 1
    artifact_policy: str = "full"
    log_every_steps: int = 100
    log_every_seconds: float = 30.0
    gmm_log_every_iters: int = 10

    def __post_init__(self):
        if isinstance(self.selected_views, str):
            self.selected_views = tuple(item for item in self.selected_views.split(",") if item)
        else:
            self.selected_views = tuple(self.selected_views)
        if self.mode == "s_only" and self.selected_views == VALID_VIEWS:
            self.selected_views = ("S",)
        if not self.selected_views:
            raise ValueError("selected_views must be a non-empty subset of S,E,P,R")
        if len(set(self.selected_views)) != len(self.selected_views) or any(v not in VALID_VIEWS for v in self.selected_views):
            raise ValueError(f"selected_views must be unique values from {VALID_VIEWS} in canonical order")
        if tuple(sorted(self.selected_views, key=VALID_VIEWS.index)) != self.selected_views:
            raise ValueError("selected_views must use canonical S,E,P,R order")
        if self.mode not in {"stage1", "full_gmm", "random_gmm", "select_gmm", "s_only", "flow"}:
            raise ValueError("mode must be stage1, full_gmm, random_gmm, select_gmm, s_only, or flow")
        if self.mode == "s_only" and self.selected_views != ("S",):
            raise ValueError("s_only mode requires selected_views=('S',) so its cache width is exactly one")
        if self.data_source not in {"large_detect", "external_detect", "synthetic"}:
            raise ValueError("data_source must be large_detect, external_detect, or synthetic")
        if self.batch_size < 1 or self.stage1_epochs < 1 or self.stage2_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, stage epochs and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.train_noise_std < 0:
            raise ValueError("learning_rate must be positive, weight_decay non-negative, and train_noise_std non-negative")
        if self.encoding_batch_size < 0 or self.cache_chunk_size < 1:
            raise ValueError("cache_chunk_size must be positive and encoding_batch_size non-negative")
        if self.encoding_batch_size and self.encoding_batch_size > self.batch_size:
            raise ValueError("encoding_batch_size cannot exceed batch_size")
        if self.gmm_components < 1 or self.gmm_fit_max_windows < 1 or self.gmm_n_init < 1 or self.gmm_max_iter < 1:
            raise ValueError("GMM component, sample, initialization, and iteration limits must be positive")
        if self.gmm_covariance_type not in {"full", "tied", "diag", "spherical"}:
            raise ValueError("gmm_covariance_type must be full, tied, diag, or spherical")
        if self.gmm_reg_covar <= 0 or self.cpu_threads < 1:
            raise ValueError("gmm_reg_covar and cpu_threads must be positive")
        if self.artifact_policy not in {"full", "results"}:
            raise ValueError("artifact_policy must be full or results")
        if self.mode == "stage1" and self.artifact_policy == "results":
            raise ValueError("stage1 requires artifact_policy=full so its source remains reusable")
        if self.log_every_steps < 1 or self.log_every_seconds <= 0 or self.gmm_log_every_iters < 1:
            raise ValueError("log intervals must be positive")

    @property
    def mnrf(self) -> MNRFConfig:
        return MNRFConfig(
            seed=self.seed, seq_len=self.seq_len, pattern_len=self.pattern_len,
            hidden_dim=self.hidden_dim, view_dim=self.view_dim,
            evolution_dim=self.evolution_dim, pattern_dim=self.pattern_dim,
            relation_dim=self.relation_dim, batch_size=self.batch_size,
            stage1_epochs=self.stage1_epochs, stage2_epochs=self.stage2_epochs,
            patience=self.patience, learning_rate=self.learning_rate,
            weight_decay=self.weight_decay, train_noise_std=self.train_noise_std,
            noise_seed=self.noise_seed, feature_std_floor=self.feature_std_floor,
            score_mad_floor=self.score_mad_floor,
            calibration_quantile=self.calibration_quantile,
            cache_chunk_size=self.cache_chunk_size,
            encoding_batch_size=self.encoding_batch_size, device=self.device,
            output_dir=self.output_dir, resume_from=self.resume_from, task_name="",
        )

    def to_dict(self):
        result = asdict(self)
        result["selected_views"] = list(self.selected_views)
        result["resolved_layout"] = self.mnrf.layout.to_dict()
        result["selected_indices"] = self.selected_indices
        result["selected_dim"] = self.selected_dim
        return result

    @property
    def layout(self):
        return self.mnrf.layout

    @property
    def selected_indices(self):
        indices = []
        for view in self.selected_views:
            start, end = self.layout.slices[view]
            indices.extend(range(start, end))
        return indices

    @property
    def selected_dim(self):
        return len(self.selected_indices)

    @classmethod
    def from_kwargs(cls, **kwargs):
        known = set(cls.__dataclass_fields__)
        unknown = set(kwargs) - known
        if unknown:
            raise TypeError("Unknown MNRFViewProbe parameters: " + ", ".join(sorted(unknown)))
        return cls(**kwargs)

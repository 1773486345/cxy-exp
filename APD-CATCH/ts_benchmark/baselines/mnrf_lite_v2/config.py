"""Frozen首版 MNRF-Lite V2 configuration."""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

from .layout import FeatureLayout


@dataclass
class MNRFConfig:
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
    feature_std_floor: float = 0.05
    train_noise_std: float = 0.01
    noise_seed: int = 20261013
    score_mad_floor: float = 0.001
    calibration_quantile: float = 0.995
    cache_chunk_size: int = 256
    encoding_batch_size: int = 0
    device: str = "auto"
    output_dir: str = ""
    resume_from: str = ""
    task_name: str = ""

    def __post_init__(self):
        # Explicit E/P/R values override the legacy common view_dim independently.
        self.evolution_dim = self.view_dim if self.evolution_dim is None else self.evolution_dim
        self.pattern_dim = self.view_dim if self.pattern_dim is None else self.pattern_dim
        self.relation_dim = self.view_dim if self.relation_dim is None else self.relation_dim

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "MNRFConfig":
        kwargs = dict(kwargs)
        saved_layout = {key: kwargs.pop(key) for key in ("order", "dimensions", "slices", "F") if key in kwargs}
        known = {field.name for field in fields(cls)}
        unknown = set(kwargs) - known
        if unknown:
            raise TypeError("Unknown MNRF-Lite V2 parameters: " + ", ".join(sorted(unknown)))
        result = cls(**kwargs)
        result.validate()
        if saved_layout and FeatureLayout.from_dict(saved_layout).to_dict() != result.layout.to_dict():
            raise ValueError("saved S/E/P/R feature layout does not match resolved configuration dimensions")
        return result

    def validate(self) -> None:
        if self.seq_len < self.pattern_len or self.pattern_len != 32:
            raise ValueError("seq_len must be >= pattern_len, and pattern_len is fixed at 32")
        if isinstance(self.view_dim, bool) or not isinstance(self.view_dim, int) or self.view_dim <= 0:
            raise ValueError("legacy view_dim must be a positive integer")
        if self.hidden_dim != 32:
            raise ValueError("hidden_dim is fixed at 32")
        self.layout
        if self.batch_size < 1 or self.stage1_epochs < 1 or self.stage2_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, stage epochs and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.cache_chunk_size < 1:
            raise ValueError("cache_chunk_size must be positive")
        if self.encoding_batch_size < 0:
            raise ValueError("encoding_batch_size must be zero (use training batch) or positive")
        if not 0.0 < self.calibration_quantile < 1.0:
            raise ValueError("calibration_quantile must be in (0,1)")

    def to_dict(self) -> Dict[str, Any]:
        # Keep this directly usable as MNRFConfig(**config.to_dict()); the
        # complete positional layout is persisted beside the config in artifacts.
        return asdict(self)

    @property
    def layout(self) -> FeatureLayout:
        return FeatureLayout(self.evolution_dim, self.pattern_dim, self.relation_dim)

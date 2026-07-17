"""Fixed decomposition scoring utilities for an unmodified CATCH instance."""

from .decomposition import (
    decompose_slow_fast,
    moving_average,
    resolve_moving_average_window,
)
from .scoring import CATCHDecompositionScorer

__all__ = [
    "CATCHDecompositionScorer",
    "decompose_slow_fast",
    "moving_average",
    "resolve_moving_average_window",
]

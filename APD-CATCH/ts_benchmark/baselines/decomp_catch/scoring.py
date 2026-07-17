"""Post-process one original CATCH reconstruction into fixed component scores."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_criterion
from ts_benchmark.baselines.utils import anomaly_detection_data_provider

from .decomposition import decompose_slow_fast, resolve_moving_average_window


EPSILON = 1e-8


class CATCHDecompositionScorer:
    """Read fixed decomposition scores from an existing original CATCH instance.

    The scorer owns no trainable state. It temporarily loads the instance's in-memory
    checkpoint when available and restores the original model state before returning.
    """

    def __init__(self, catch: Any):
        if getattr(catch, "model", None) is None:
            raise ValueError("CATCHDecompositionScorer requires an initialized CATCH model")
        if not hasattr(catch, "config") or not hasattr(catch, "scaler"):
            raise TypeError("catch must expose the original CATCH config and scaler")
        self.catch = catch
        self.window_size = resolve_moving_average_window(
            int(catch.config.patch_size), int(catch.config.seq_len)
        )

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Expose an empty iterator so callers can verify no trainable state exists."""
        return iter(())

    def fit_normalization_stats(
        self,
        reference_data: pd.DataFrame,
        source_name: str,
        scoring_seed: int = 20260717,
    ) -> Dict[str, Any]:
        """Fit fixed slow/fast z-score statistics from an allowed reference segment."""
        if source_name not in {"validation", "train"}:
            raise ValueError("source_name must be 'validation' or 'train'")
        result = self._score_without_fusion(reference_data, scoring_seed)
        slow_scale = float(np.std(result["slow_score"]))
        fast_scale = float(np.std(result["fast_score"]))
        return {
            "slow_location": float(np.mean(result["slow_score"])),
            "slow_scale": slow_scale,
            "fast_location": float(np.mean(result["fast_score"])),
            "fast_scale": fast_scale,
            "epsilon": EPSILON,
            "source": source_name,
            "source_length": result["source_length"],
            "scored_length": result["scored_length"],
            "dropped_tail_length": result["dropped_tail_length"],
            "scoring_window_size": result["scoring_window_size"],
            "window_step": result["window_step"],
            "time_index_alignment": result["time_index_alignment"],
            "scoring_seed": result["scoring_seed"],
            "zero_scale_diagnostics": {
                "slow_scale_is_zero": slow_scale == 0.0,
                "fast_scale_is_zero": fast_scale == 0.0,
            },
        }

    def score_dataframe(
        self,
        data: pd.DataFrame,
        normalization_stats: Optional[Dict[str, Any]] = None,
        scoring_seed: int = 20260717,
    ) -> Dict[str, Any]:
        """Score non-overlapping CATCH windows with one reconstruction forward each."""
        if normalization_stats is None:
            raise ValueError(
                "normalization_stats is required; fit it from validation or train data first"
            )
        stats = self._validate_normalization_stats(normalization_stats)
        result = self._score_without_fusion(data, scoring_seed)
        slow_z = (result["slow_score"] - stats["slow_location"]) / (
            stats["slow_scale"] + EPSILON
        )
        fast_z = (result["fast_score"] - stats["fast_location"]) / (
            stats["fast_scale"] + EPSILON
        )
        result["fusion_score"] = 0.5 * slow_z + 0.5 * fast_z
        result["normalization_stats"] = copy.deepcopy(stats)
        return result

    def _score_without_fusion(
        self, data: pd.DataFrame, scoring_seed: int
    ) -> Dict[str, Any]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("data must not be empty")
        if not isinstance(scoring_seed, int) or isinstance(scoring_seed, bool):
            raise TypeError("scoring_seed must be an integer")

        config = self.catch.config
        sequence_length = int(config.seq_len)
        source_length = len(data)
        scored_length = (source_length // sequence_length) * sequence_length
        if scored_length == 0:
            raise ValueError("data must contain at least one complete CATCH scoring window")
        dropped_tail_length = source_length - scored_length

        scaled = pd.DataFrame(
            self.catch.scaler.transform(data.values),
            columns=data.columns,
            index=data.index,
        )
        loader = anomaly_detection_data_provider(
            scaled,
            batch_size=int(config.batch_size),
            win_size=sequence_length,
            step=1,
            mode="thre",
        )

        model = self.catch.model
        parameter = next(model.parameters(), None)
        if parameter is None:
            raise ValueError("CATCH model has no parameters")
        device = parameter.device
        uses_cuda = device.type == "cuda"
        frequency = frequency_criterion(config).to(device)
        checkpoint, checkpoint_source = self._checkpoint_state(model)
        original_state = copy.deepcopy(model.state_dict())
        was_training = model.training

        score_lists = {
            "original_score": [],
            "time_score": [],
            "frequency_score": [],
            "slow_score": [],
            "fast_score": [],
        }
        decomposition_error = 0.0
        forward_calls = 0

        with self._controlled_rng(scoring_seed, uses_cuda) as rng_metadata:
            try:
                model.load_state_dict(checkpoint)
                model.eval()
                with torch.no_grad():
                    for batch_x, _ in loader:
                        batch_x = batch_x.float().to(device)
                        x_hat, _, _ = model(batch_x)
                        forward_calls += 1

                        time_score = torch.mean((batch_x - x_hat) ** 2, dim=-1)
                        frequency_score = torch.mean(
                            frequency(batch_x, x_hat), dim=-1
                        )
                        original_score = (
                            time_score + float(config.score_lambda) * frequency_score
                        )

                        slow_x, fast_x = decompose_slow_fast(batch_x, self.window_size)
                        slow_hat, fast_hat = decompose_slow_fast(x_hat, self.window_size)
                        slow_score = torch.mean((slow_x - slow_hat) ** 2, dim=-1)
                        fast_score = torch.mean((fast_x - fast_hat) ** 2, dim=-1)

                        decomposition_error = max(
                            decomposition_error,
                            float(torch.max(torch.abs(slow_x + fast_x - batch_x)).item()),
                            float(torch.max(torch.abs(slow_hat + fast_hat - x_hat)).item()),
                        )
                        score_lists["original_score"].append(original_score.cpu())
                        score_lists["time_score"].append(time_score.cpu())
                        score_lists["frequency_score"].append(frequency_score.cpu())
                        score_lists["slow_score"].append(slow_score.cpu())
                        score_lists["fast_score"].append(fast_score.cpu())
            finally:
                model.load_state_dict(original_state)
                model.train(was_training)

        flattened = {
            name: torch.cat(values, dim=0).reshape(-1).numpy()
            for name, values in score_lists.items()
        }
        if any(len(values) != scored_length for values in flattened.values()):
            raise RuntimeError("CATCH score expansion did not cover complete scoring windows")

        return {
            **flattened,
            "time_index": np.asarray(data.index[:scored_length]),
            "source_length": source_length,
            "scored_length": scored_length,
            "dropped_tail_length": dropped_tail_length,
            "window_size": self.window_size,
            "scoring_window_size": sequence_length,
            "window_step": sequence_length,
            "time_index_alignment": "flattened non-overlapping windows from input index 0",
            "scoring_seed": scoring_seed,
            "cpu_rng_control": rng_metadata["cpu_rng_control"],
            "cuda_rng_control": rng_metadata["cuda_rng_control"],
            "uses_cuda": uses_cuda,
            "checkpoint_source": checkpoint_source,
            "forward_calls": forward_calls,
            "decomposition_reconstruction_max_error": decomposition_error,
        }

    def _checkpoint_state(
        self, model: torch.nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        early_stopping = getattr(self.catch, "early_stopping", None)
        checkpoint = getattr(early_stopping, "check_point", None)
        if checkpoint is None:
            return copy.deepcopy(model.state_dict()), "current_model_state"
        return copy.deepcopy(checkpoint), "catch.early_stopping.check_point"

    @staticmethod
    def _validate_normalization_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
        required = {
            "slow_location",
            "slow_scale",
            "fast_location",
            "fast_scale",
            "epsilon",
            "source",
        }
        missing = required - set(stats)
        if missing:
            raise ValueError(f"normalization_stats is missing: {sorted(missing)}")
        if stats["source"] not in {"validation", "train"}:
            raise ValueError("normalization_stats source must be 'validation' or 'train'")
        if float(stats["epsilon"]) != EPSILON:
            raise ValueError("normalization_stats epsilon must equal 1e-8")
        return copy.deepcopy(stats)

    @staticmethod
    @contextmanager
    def _controlled_rng(scoring_seed: int, uses_cuda: bool) -> Iterator[Dict[str, str]]:
        cpu_state = torch.random.get_rng_state()
        cuda_available = torch.cuda.is_available()
        cuda_states = torch.cuda.get_rng_state_all() if cuda_available else None
        torch.manual_seed(scoring_seed)
        if cuda_available:
            torch.cuda.manual_seed_all(scoring_seed)
        try:
            yield {
                "cpu_rng_control": "saved, torch.manual_seed(scoring_seed), restored",
                "cuda_rng_control": (
                    "saved, torch.cuda.manual_seed_all(scoring_seed), restored"
                    if cuda_available
                    else "CUDA unavailable; no CUDA RNG state used"
                ),
                "uses_cuda": str(uses_cuda),
            }
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

"""Persistent, label-free interventions for single-stage v2 training."""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn


TYPE_NAMES = ("state", "evolution", "pattern", "relation")
PAIR_TYPES = tuple(itertools.combinations(range(4), 2))


def _randint(generator: torch.Generator, low: int, high: int, device: torch.device) -> int:
    return int(torch.randint(low, max(low + 1, high), (), generator=generator, device=device).item())


def _uniform(generator: torch.Generator, low: float, high: float, device: torch.device) -> float:
    return float(torch.empty((), device=device).uniform_(low, high, generator=generator).item())


class TypeInterventionGenerator(nn.Module):
    def __init__(self, seed: int = 2021) -> None:
        super().__init__()
        if not isinstance(seed, int):
            seed = int(getattr(seed, "seed", 2021))
        self.seed = int(seed)
        self._training_generators: Dict[str, torch.Generator] = {}

    def _new_generator(self, seed: int, device: torch.device) -> torch.Generator:
        generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
        generator.manual_seed(int(seed))
        return generator

    def _training_generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._training_generators:
            self._training_generators[key] = self._new_generator(self.seed, device)
        return self._training_generators[key]

    @staticmethod
    def _interval(length: int, low: float, high: float, generator: torch.Generator, device: torch.device) -> Tuple[int, int]:
        span_low = max(1, int(round(length * low)))
        span_high = max(span_low + 1, int(round(length * high)) + 1)
        span = min(length, _randint(generator, span_low, span_high, device))
        start = _randint(generator, 0, max(1, length - span + 1), device)
        return start, min(length, start + span)

    @staticmethod
    def _channels(count: int, low: float, high: float, generator: torch.Generator, device: torch.device) -> Tensor:
        lower = max(1, int(round(count * low)))
        upper = max(lower + 1, int(round(count * high)) + 1)
        return torch.randperm(count, generator=generator, device=device)[:min(count, _randint(generator, lower, upper, device))]

    def _sample_params(self, original: Tensor, donor: Tensor, kind: int, generator: torch.Generator, weak: bool, interval: Optional[Tuple[int, int]] = None) -> Dict[str, object]:
        time, channels = original.shape
        params: Dict[str, object] = {"kind": kind, "weak": weak}
        if kind == 0:
            params["interval"] = interval or self._interval(time, 0.10, 0.30, generator, original.device)
            params["channels"] = self._channels(channels, 0.20, 0.50, generator, original.device)
            params["amount"] = _uniform(generator, 0.15 if weak else 0.75, 0.35 if weak else 1.50, original.device) * (-1 if _randint(generator, 0, 2, original.device) == 0 else 1)
        elif kind == 1:
            params["interval"] = interval or self._interval(time, 0.15, 0.35, generator, original.device)
            params["alpha"] = _uniform(generator, 0.20, 0.40, original.device) if weak else 1.0
        elif kind == 2:
            params["interval"] = interval or self._interval(time, 0.25, 0.50, generator, original.device)
            start, end = params["interval"]
            parts = min(3 if _randint(generator, 0, 2, original.device) == 0 else 4, max(1, end - start))
            permutation = torch.randperm(parts, generator=generator, device=original.device)
            identity = torch.arange(parts, device=original.device)
            if parts > 1 and torch.equal(permutation, identity):
                permutation = torch.roll(permutation, 1)
            params["parts"] = parts
            params["permutation"] = permutation
            params["alpha"] = _uniform(generator, 0.20, 0.40, original.device) if weak else 1.0
        else:
            params["interval"] = interval or self._interval(time, 0.15, 0.35, generator, original.device)
            params["channels"] = self._channels(channels, 0.10, 0.35, generator, original.device)
            params["alpha"] = _uniform(generator, 0.20, 0.40, original.device) if weak else 1.0
        return params

    def _apply_intervention(self, original: Tensor, donor: Tensor, params: Dict[str, object]) -> Tuple[Tensor, Tensor]:
        result = original.clone()
        time, channels = original.shape
        start, end = params["interval"]
        kind = int(params["kind"])
        mask = torch.zeros(time, channels, dtype=torch.bool, device=original.device)
        if kind == 0:
            selected = params["channels"]
            result[start:end, selected] += float(params["amount"])
            mask[start:end, selected] = True
        elif kind == 1:
            aligned = donor[start:end].clone()
            if start > 0 and aligned.numel():
                aligned = aligned + (original[start - 1] - aligned[0]).unsqueeze(0)
            alpha = float(params["alpha"])
            result[start:end] = (1.0 - alpha) * result[start:end] + alpha * aligned
            mask[start:end] = True
        elif kind == 2:
            parts = int(params["parts"])
            permutation = params["permutation"]
            length = end - start
            bounds = torch.linspace(0, length, parts + 1, dtype=torch.long, device=original.device)
            chunks = [original[start + int(bounds[i]):start + int(bounds[i + 1])] for i in range(parts)]
            permuted = torch.cat([chunks[int(index)] for index in permutation], dim=0)
            alpha = float(params["alpha"])
            result[start:end] = (1.0 - alpha) * result[start:end] + alpha * permuted
            mask[start:end] = True
        else:
            selected_channels = params["channels"]
            source = donor[start:end, selected_channels]
            target = original[start:end, selected_channels]
            source_mean = source.mean(dim=0, keepdim=True)
            source_std = source.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
            target_std = target.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
            matched = (source - source_mean) / source_std * target_std + target.mean(dim=0, keepdim=True)
            alpha = float(params["alpha"])
            result[start:end, selected_channels] = (1.0 - alpha) * target + alpha * matched
            mask[start:end, selected_channels] = True
        return result, mask

    def generate(self, x: Tensor, *, seed: Optional[int] = None, sample_indices: Optional[Iterable[int]] = None, validation: bool = False, generator: Optional[torch.Generator] = None) -> Dict[str, Tensor]:
        if x.ndim != 3:
            raise ValueError("x must have shape [B,T,C]")
        batch, time, channels = x.shape
        indices = list(sample_indices) if sample_indices is not None else list(range(batch))
        if len(indices) != batch:
            raise ValueError("sample_indices must contain one index per batch item")
        out = x.clone()
        targets = torch.zeros(batch, 4, dtype=x.dtype, device=x.device)
        masks = torch.zeros(batch, 4, time, channels, dtype=torch.bool, device=x.device)
        scenario = torch.zeros(batch, dtype=torch.long, device=x.device)
        weak_i = torch.zeros_like(x)
        weak_j = torch.zeros_like(x)
        weak_compound = torch.zeros_like(x)
        weak_mask_i = torch.zeros(batch, time, dtype=torch.bool, device=x.device)
        weak_mask_j = torch.zeros_like(weak_mask_i)
        donor_indices = torch.full((batch,), -1, dtype=torch.long, device=x.device)
        debug_intervals = torch.full((batch, 4, 2), -1, dtype=torch.long, device=x.device)
        debug_channels = torch.zeros(batch, 4, channels, dtype=torch.bool, device=x.device)
        debug_strengths = torch.zeros(batch, 4, dtype=x.dtype, device=x.device)
        for b in range(batch):
            if validation:
                local_seed = (self.seed if seed is None else int(seed)) + int(indices[b])
                gen = self._new_generator(local_seed, x.device)
            else:
                gen = generator or self._training_generator(x.device)
            if batch == 1:
                donor_indices[b] = b
                donor = torch.roll(x[b], shifts=1, dims=0)
            else:
                # Draw an offset from [1, batch-1], then wrap the local sample
                # index.  This makes self-donation impossible for every batch.
                offset = _randint(gen, 1, batch, x.device)
                sample_index = b
                donor_index = (sample_index + offset) % batch
                if donor_index == b:
                    raise RuntimeError("intervention donor selection produced self-donation")
                donor_indices[b] = donor_index
                donor = x[donor_index]
            draw = _uniform(gen, 0.0, 1.0, x.device)
            if draw < 0.25:
                continue
            if draw < 0.75:
                scenario[b] = 1
                kinds = (_randint(gen, 0, 4, x.device),)
                weak = False
            elif draw < 0.875:
                scenario[b] = 2
                kinds = PAIR_TYPES[_randint(gen, 0, len(PAIR_TYPES), x.device)]
                weak = False
            else:
                scenario[b] = 3
                kinds = PAIR_TYPES[_randint(gen, 0, len(PAIR_TYPES), x.device)]
                weak = True
            shared_interval = self._interval(time, 0.15, 0.35, gen, x.device) if weak else None
            base = x[b]
            sampled = []
            for kind in kinds:
                params = self._sample_params(base, donor, kind, gen, weak, shared_interval)
                sampled.append(params)
                base, mask = self._apply_intervention(base, donor, params)
                targets[b, kind] = 1.0
                masks[b, kind] = mask
                start, end = params["interval"]
                debug_intervals[b, kind] = torch.tensor((start, end), dtype=torch.long, device=x.device)
                debug_channels[b, kind] = mask.any(dim=0)
                if kind == 0:
                    debug_strengths[b, kind] = abs(float(params["amount"]))
                elif "alpha" in params:
                    debug_strengths[b, kind] = abs(float(params["alpha"]))
                else:
                    debug_strengths[b, kind] = 1.0
            out[b] = base
            if weak:
                first, second = sampled
                weak_i_value, weak_i_value_mask = self._apply_intervention(x[b], donor, first)
                weak_j_value, weak_j_value_mask = self._apply_intervention(x[b], donor, second)
                weak_i[b] = weak_i_value
                weak_j[b] = weak_j_value
                weak_mask_i[b] = weak_i_value_mask.any(dim=-1)
                weak_mask_j[b] = weak_j_value_mask.any(dim=-1)
                weak_compound[b], _ = self._apply_intervention(weak_i[b], donor, second)
        union = masks.any(dim=1).any(dim=-1)
        return {
            "corrupted_x": out,
            "type_targets": targets,
            "type_masks": masks,
            "union_mask": union,
            "scenario_kind": scenario,
            "weak_view_i": weak_i,
            "weak_view_j": weak_j,
            "weak_compound_view": weak_compound,
            "weak_mask_i": weak_mask_i,
            "weak_mask_j": weak_mask_j,
            "donor_indices": donor_indices,
            "debug_donor_indices": donor_indices,
            "debug_intervals": debug_intervals,
            "debug_channels": debug_channels,
            "debug_strengths": debug_strengths,
        }

    forward = generate
    sample = generate


TypeInterventions = TypeInterventionGenerator
TypeIntervention = TypeInterventionGenerator
InterventionGenerator = TypeInterventionGenerator

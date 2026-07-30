"""Deterministic, label-free normal-window interventions for Phase B."""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn


TYPE_NAMES = ("state", "evolution", "pattern", "relation")
PAIR_TYPES = tuple(itertools.combinations(range(4), 2))


def _randint(generator: torch.Generator, low: int, high: int, device: torch.device) -> int:
    if high <= low:
        return int(low)
    return int(torch.randint(low, high, (), generator=generator, device=device).item())


def _uniform(generator: torch.Generator, low: float, high: float, device: torch.device) -> float:
    return float(torch.empty((), device=device).uniform_(low, high, generator=generator).item())


class TypeInterventionGenerator(nn.Module):
    """Generate clean/single/compound views using only a torch Generator.

    ``validation=True`` derives one local generator from ``seed + sample_index``
    for each item, making validation interventions independent of batch order.
    """

    def __init__(self, seed: int = 2021) -> None:
        super().__init__()
        self.seed = int(getattr(seed, "seed", seed))

    def _generator(self, seed: Optional[int], device: torch.device) -> torch.Generator:
        # A CPU generator is accepted for CPU and CUDA random draws and avoids
        # dependence on the process-global RNG state.
        generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
        generator.manual_seed(self.seed if seed is None else int(seed))
        return generator

    @staticmethod
    def _interval(length: int, low: float, high: float, generator: torch.Generator, device: torch.device) -> Tuple[int, int]:
        span_low = max(1, int(round(length * low)))
        span_high = max(span_low + 1, int(round(length * high)) + 1)
        span = min(length, _randint(generator, span_low, span_high, device))
        start = _randint(generator, 0, max(1, length - span + 1), device)
        return start, min(length, start + span)

    @staticmethod
    def _channel_subset(channels: int, low: float, high: float, generator: torch.Generator, device: torch.device) -> Tensor:
        count_low = max(1, int(round(channels * low)))
        count_high = max(count_low + 1, int(round(channels * high)) + 1)
        count = min(channels, _randint(generator, count_low, count_high, device))
        return torch.randperm(channels, generator=generator, device=device)[:count]

    def _state(self, original: Tensor, generator: torch.Generator, weak: bool) -> Tuple[Tensor, Tensor]:
        t, c = original.shape
        shared = getattr(self, "_weak_interval", None) if weak else None
        start, end = shared or self._interval(t, 0.10, 0.30, generator, original.device)
        channels = self._channel_subset(c, 0.20, 0.50, generator, original.device)
        amount = _uniform(generator, 0.15 if weak else 0.75, 0.35 if weak else 1.50, original.device)
        amount *= -1.0 if _randint(generator, 0, 2, original.device) == 0 else 1.0
        result = original.clone()
        result[start:end, channels] += amount
        mask = torch.zeros((t, c), dtype=torch.bool, device=original.device)
        mask[start:end, channels] = True
        return result, mask

    def _evolution(self, original: Tensor, donor: Tensor, generator: torch.Generator, weak: bool) -> Tuple[Tensor, Tensor]:
        t, _ = original.shape
        shared = getattr(self, "_weak_interval", None) if weak else None
        start, end = shared or self._interval(t, 0.15, 0.35, generator, original.device)
        # Preserve continuity at the transition by shifting donor's first point.
        aligned = donor[start:end].clone()
        if start > 0 and aligned.shape[0] > 0:
            aligned = aligned + (original[start - 1] - aligned[0]).unsqueeze(0)
        result = original.clone()
        if weak:
            alpha = _uniform(generator, 0.20, 0.40, original.device)
            result[start:end] = (1.0 - alpha) * result[start:end] + alpha * aligned
        else:
            result[start:end] = aligned
        mask = torch.zeros((t, original.shape[-1]), dtype=torch.bool, device=original.device)
        mask[start:end] = True
        return result, mask

    def _pattern(self, original: Tensor, generator: torch.Generator, weak: bool) -> Tuple[Tensor, Tensor]:
        t, c = original.shape
        shared = getattr(self, "_weak_interval", None) if weak else None
        start, end = shared or self._interval(t, 0.25, 0.50, generator, original.device)
        length = end - start
        n_parts = 3 if _randint(generator, 0, 2, original.device) == 0 else 4
        n_parts = min(n_parts, max(1, length))
        bounds = torch.linspace(0, length, n_parts + 1, dtype=torch.long, device=original.device)
        # Draw until a non-identity permutation (for n_parts >= 2).
        perm = torch.randperm(n_parts, generator=generator, device=original.device)
        if n_parts > 1 and torch.equal(perm, torch.arange(n_parts, device=original.device)):
            perm = torch.roll(perm, 1, 0)
        chunks = [original[start + int(bounds[i].item()): start + int(bounds[i + 1].item())] for i in range(n_parts)]
        permuted = torch.cat([chunks[int(i)] for i in perm], dim=0) if chunks else original[start:end]
        result = original.clone()
        if weak:
            alpha = _uniform(generator, 0.20, 0.40, original.device)
            result[start:end] = (1.0 - alpha) * result[start:end] + alpha * permuted
        else:
            result[start:end] = permuted
        mask = torch.zeros((t, c), dtype=torch.bool, device=original.device)
        mask[start:end] = True
        return result, mask

    def _relation(self, original: Tensor, donor: Tensor, generator: torch.Generator, weak: bool) -> Tuple[Tensor, Tensor]:
        t, c = original.shape
        shared = getattr(self, "_weak_interval", None) if weak else None
        start, end = shared or self._interval(t, 0.15, 0.35, generator, original.device)
        channels = self._channel_subset(c, 0.10, 0.35, generator, original.device)
        result = original.clone()
        selected = donor[start:end, channels]
        target = original[start:end, channels]
        mean = selected.mean(dim=0, keepdim=True)
        std = selected.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
        target_std = target.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
        matched = (selected - mean) / std * target_std + target.mean(dim=0, keepdim=True)
        if weak:
            alpha = _uniform(generator, 0.20, 0.40, original.device)
            matched = (1.0 - alpha) * target + alpha * matched
        result[start:end, channels] = matched
        mask = torch.zeros((t, c), dtype=torch.bool, device=original.device)
        mask[start:end, channels] = True
        return result, mask

    def _one_type(self, original: Tensor, donor: Tensor, kind: int, generator: torch.Generator, weak: bool) -> Tuple[Tensor, Tensor]:
        if kind == 0:
            return self._state(original, generator, weak)
        if kind == 1:
            return self._evolution(original, donor, generator, weak)
        if kind == 2:
            return self._pattern(original, generator, weak)
        return self._relation(original, donor, generator, weak)

    def generate(
        self,
        x: Tensor,
        *,
        seed: Optional[int] = None,
        sample_indices: Optional[Iterable[int]] = None,
        validation: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Tensor]:
        if x.ndim != 3:
            raise ValueError("x must have shape [B,T,C]")
        batch, time, channels = x.shape
        indices = list(sample_indices) if sample_indices is not None else list(range(batch))
        if len(indices) != batch:
            raise ValueError("sample_indices must contain one index per batch item")
        out = x.clone()
        targets = torch.zeros((batch, 4), dtype=x.dtype, device=x.device)
        masks = torch.zeros((batch, 4, time, channels), dtype=torch.bool, device=x.device)
        scenario = torch.zeros(batch, dtype=torch.long, device=x.device)
        weak_views = {"weak_view_i": torch.zeros_like(x), "weak_view_j": torch.zeros_like(x), "weak_compound_view": torch.zeros_like(x)}
        shared_generator = None
        if not validation and generator is None:
            shared_generator = self._generator(seed, x.device)
        for b in range(batch):
            local_seed = (self.seed if seed is None else int(seed)) + int(indices[b]) if validation else seed
            if validation:
                gen = self._generator(local_seed, x.device)
            else:
                gen = generator if generator is not None else shared_generator
            draw = _uniform(gen, 0.0, 1.0, x.device)
            if draw < 0.25:
                scenario[b] = 0  # clean
                continue
            if draw < 0.75:
                scenario[b] = 1  # single strong
                kinds = (_randint(gen, 0, 4, x.device),)
            elif draw < 0.875:
                scenario[b] = 2  # compound strong
                kinds = PAIR_TYPES[_randint(gen, 0, len(PAIR_TYPES), x.device)]
            else:
                scenario[b] = 3  # compound weak
                kinds = PAIR_TYPES[_randint(gen, 0, len(PAIR_TYPES), x.device)]
            donor_index = _randint(gen, 0, batch, x.device)
            if batch > 1 and donor_index == b:
                donor_index = (donor_index + 1) % batch
            donor = x[donor_index]
            base = x[b]
            if scenario[b].item() == 3:
                self._weak_interval = self._interval(time, 0.15, 0.35, gen, x.device)
            for kind in kinds:
                view, mask = self._one_type(base, donor, int(kind), gen, weak=(scenario[b].item() == 3))
                # Compound interventions compose on the same working view.
                base = view
                out[b] = view
                targets[b, kind] = 1.0
                masks[b, kind] |= mask
            if scenario[b].item() == 3 and len(kinds) == 2:
                # Auxiliary views are only materialized for weak compounds.
                i, j = kinds
                weak_views["weak_view_i"][b], _ = self._one_type(x[b], donor, i, gen, weak=True)
                weak_views["weak_view_j"][b], _ = self._one_type(x[b], donor, j, gen, weak=True)
                weak_views["weak_compound_view"][b] = out[b]
            if scenario[b].item() == 3:
                self._weak_interval = None
        union = masks.any(dim=1).any(dim=-1)
        result: Dict[str, Tensor] = {
            "corrupted_x": out,
            "type_targets": targets,
            "type_masks": masks,
            "union_mask": union,
            "scenario_kind": scenario,
        }
        result.update(weak_views)
        return result

    forward = generate
    sample = generate


TypeInterventions = TypeInterventionGenerator
TypeIntervention = TypeInterventionGenerator
InterventionGenerator = TypeInterventionGenerator

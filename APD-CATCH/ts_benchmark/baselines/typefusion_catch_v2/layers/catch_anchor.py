"""Frozen normal-context anchor around the original CATCH model.

The wrapper deliberately delegates the numerical forward pass to ``CATCHModel``.
This keeps Phase A checkpoint compatibility and gives Phase B a small, explicit
interface without copying any of the original implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn


class CATCHAnchor(nn.Module):
    """Expose CATCH's reconstruction and frequency outputs as anchor tensors.

    Parameters
    ----------
    catch_model:
        An already constructed ``ts_benchmark.baselines.catch.models.CATCHModel``.
    configs:
        Optional CATCH config used to construct the model when ``catch_model`` is
        omitted.  Supplying an existing model is preferred for strict state-dict
        parity tests.
    """

    def __init__(self, catch_model: Optional[nn.Module] = None, configs: Any = None, **kwargs: Any) -> None:
        super().__init__()
        if catch_model is not None and not isinstance(catch_model, nn.Module) and configs is None:
            configs, catch_model = catch_model, None
        if catch_model is None:
            if configs is None:
                catch_model = kwargs.pop("model", None)
            if catch_model is None and configs is not None:
                from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel

                catch_model = CATCHModel(configs)
        if catch_model is None:
            raise ValueError("CATCHAnchor requires catch_model or configs")
        self.catch_model = catch_model

    @property
    def model(self) -> nn.Module:
        """Compatibility alias used by adapters and checkpoint code."""

        return self.catch_model

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True):
        """Accept either an anchor wrapper state or an original CATCH state.

        Phase A checkpoints contain the original model key space, whereas a
        wrapper checkpoint naturally prefixes keys with ``catch_model.``.  Both
        forms are unambiguous and are loaded strictly by default.
        """

        if state_dict and not any(key.startswith("catch_model.") for key in state_dict):
            return self.catch_model.load_state_dict(state_dict, strict=strict)
        return super().load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        result = self.catch_model(x)
        if isinstance(result, dict):
            reconstruction = result.get("anchor_reconstruction", result.get("reconstruction"))
            spectrum = result.get("anchor_spectrum", result.get("spectrum"))
            dc_loss = result.get("anchor_dc_loss", result.get("dc_loss"))
        elif isinstance(result, (tuple, list)):
            if len(result) < 3:
                raise RuntimeError("CATCH forward must return reconstruction, spectrum and dc loss")
            reconstruction, spectrum, dc_loss = result[:3]
        else:
            raise RuntimeError("unsupported CATCH forward result type")
        if reconstruction is None or spectrum is None or dc_loss is None:
            raise RuntimeError("CATCH forward did not provide required anchor outputs")

        # CATCH's frequency output is [B,T,C] complex.  Keep it untouched for
        # parity; a real latent view is supplied only as context input.
        if torch.is_complex(spectrum):
            frequency_latent = torch.cat((spectrum.real, spectrum.imag), dim=-1)
        else:
            frequency_latent = spectrum
        return {
            "anchor_reconstruction": reconstruction,
            "anchor_spectrum": spectrum,
            "anchor_dc_loss": dc_loss,
            "anchor_frequency_latent": frequency_latent,
        }

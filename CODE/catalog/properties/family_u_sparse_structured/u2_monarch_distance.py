"""
u2_monarch_distance.py — Property U2 : distance à Monarch (V1 mask).

Spec : DOC/CATALOGUE §U2 "Monarch".

V1 : projection sur mask Monarch (m=√N, b=√N par défaut). Identifie la
fraction d'énergie capturée par le support théorique.

V2 : factorisation ALS Monarch (deux étages PDPD).
"""

from __future__ import annotations

import torch

from catalog.projectors import MonarchMask
from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class U2MonarchDistance(Property):
    """U2 — borne inf ε_monarch via mask Monarch."""

    name = "U2_monarch_distance"
    family = "U"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self) -> None:
        self._projector = MonarchMask()

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        eps = self._projector.epsilon(A_work)
        eps_flat = eps.float().flatten()

        mask = self._projector._get_mask(N, A_work.device, A_work.dtype)
        density = float(mask.float().mean().item())

        return {
            "epsilon_monarch_lb_median": float(eps_flat.median().item()),
            "epsilon_monarch_lb_mean": float(eps_flat.mean().item()),
            "epsilon_monarch_lb_min": float(eps_flat.min().item()),
            "fraction_close_to_monarch_below_0p30": float(
                (eps_flat < 0.30).float().mean().item()
            ),
            "mask_density": density,
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }

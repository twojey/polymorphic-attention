"""
u4_pixelfly_distance.py — Property U4 : distance Pixelfly (V1 mask).

Spec : DOC/CATALOGUE §U4.

Pixelfly est un pattern hand-designed pour attention sparse : il combine
- diagonal (local attention)
- random global (sparse global)
- block-local

V1 simple : on construit le pattern Pixelfly classique avec 3 composants :
- diagonal band ±2
- 4 entrées random globales par ligne (seed fixe)
- 1 bloc local 4×4 dans le coin

Puis projection sur ce mask.
"""

from __future__ import annotations

import torch

from catalog.projectors.base import Projector
from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _pixelfly_mask(n: int, band: int, n_global: int, block_size: int, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Construit le mask Pixelfly hand-designed."""
    mask = torch.zeros(n, n, device=device, dtype=torch.bool)
    # 1. Bande diagonale ±band
    for d in range(-band, band + 1):
        for i in range(n):
            j = i + d
            if 0 <= j < n:
                mask[i, j] = True
    # 2. Random global per row
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    for i in range(n):
        cols = torch.randperm(n, generator=gen)[:n_global]
        for j in cols.tolist():
            mask[i, j] = True
    # 3. Bloc local en coin
    if block_size > 0 and block_size <= n:
        mask[:block_size, :block_size] = True
    return mask.to(dtype=dtype)


class PixelflyMask(Projector):
    """Projection sur mask Pixelfly hand-designed."""

    name = "pixelfly_mask"
    family = "U"

    def __init__(self, band: int = 2, n_global: int = 4, block_size: int = 4, seed: int = 0) -> None:
        self.band = band
        self.n_global = n_global
        self.block_size = block_size
        self.seed = seed
        self._cache: dict[tuple[int, str, str], torch.Tensor] = {}

    def _get_mask(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (n, str(device), str(dtype))
        if key not in self._cache:
            self._cache[key] = _pixelfly_mask(
                n, self.band, self.n_global, self.block_size, self.seed, device, dtype
            )
        return self._cache[key]

    def project(self, A: torch.Tensor) -> torch.Tensor:
        mask = self._get_mask(A.shape[-1], A.device, A.dtype)
        return A * mask


@register_property
class U4PixelflyDistance(Property):
    """U4 — distance Frobenius au mask Pixelfly."""

    name = "U4_pixelfly_distance"
    family = "U"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self) -> None:
        self._projector = PixelflyMask()

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
            "epsilon_pixelfly_median": float(eps_flat.median().item()),
            "epsilon_pixelfly_mean": float(eps_flat.mean().item()),
            "fraction_close_to_pixelfly_below_0p30": float(
                (eps_flat < 0.30).float().mean().item()
            ),
            "mask_density": density,
            "n_matrices": int(B * H),
        }

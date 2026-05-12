"""
l2_haar_wavelets.py — Property L2 : analyse multi-échelle Haar wavelets.

Spec : DOC/CATALOGUE §L2 "wavelets (Daubechies, Haar)".

Implémentation simple : transformée Haar 1D appliquée par ligne (puis
par colonne sur transposée pour analyse 2D). Métriques par niveau :
- Energy par bande (LL, LH, HL, HH au niveau 1 ; cascade au niveau 2)
- Sparsité par bande (fraction de coefficients négligeables)

V1 : 2 niveaux Haar (suffisant pour caractériser concentration multi-échelle).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _haar_1d_along_dim(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Décomposition Haar 1D sur la dimension `dim`.

    Retourne (cA, cD) — approximation et détail. Si len(dim) est impair, on
    pad d'un 0 à la fin.
    """
    n = x.shape[dim]
    if n % 2 == 1:
        # Pad un zéro à la fin sur la dimension cible
        pad_shape = list(x.shape)
        pad_shape[dim] = 1
        zero = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
        x = torch.cat([x, zero], dim=dim)
        n += 1
    # Extraire pairs et impairs
    even = x.index_select(dim, torch.arange(0, n, 2, device=x.device))
    odd = x.index_select(dim, torch.arange(1, n, 2, device=x.device))
    cA = (even + odd) / 2.0
    cD = (even - odd) / 2.0
    return cA, cD


@register_property
class L2HaarWavelets(Property):
    """L2 — décomposition Haar 2D 2-niveaux et énergies par bande."""

    name = "L2_haar_wavelets"
    family = "L"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Décomposition Haar 2D niveau 1 : lignes puis colonnes
        # Sur dim=-1 (colonnes)
        cAc, cDc = _haar_1d_along_dim(A_work, dim=-1)  # (B, H, N, N/2) chacun
        # Puis sur dim=-2 (rows) appliqué à cAc et cDc
        LL, LH = _haar_1d_along_dim(cAc, dim=-2)  # LL = approximation, LH = horizontal détail
        HL, HH = _haar_1d_along_dim(cDc, dim=-2)  # HL = vertical, HH = diagonal

        # Énergies par bande, normalisées par énergie totale
        def _energy(t: torch.Tensor) -> torch.Tensor:
            return t.pow(2).flatten(start_dim=-2).sum(dim=-1)

        e_LL = _energy(LL)
        e_LH = _energy(LH)
        e_HL = _energy(HL)
        e_HH = _energy(HH)
        total = (e_LL + e_LH + e_HL + e_HH).clamp_min(1e-30)

        ll_f = (e_LL / total).float().flatten()
        lh_f = (e_LH / total).float().flatten()
        hl_f = (e_HL / total).float().flatten()
        hh_f = (e_HH / total).float().flatten()

        results: dict[str, float | int | str | bool] = {
            "energy_LL_fraction_median": float(ll_f.median().item()),
            "energy_LH_fraction_median": float(lh_f.median().item()),
            "energy_HL_fraction_median": float(hl_f.median().item()),
            "energy_HH_fraction_median": float(hh_f.median().item()),
            "energy_LL_fraction_mean": float(ll_f.mean().item()),
            "energy_high_freq_fraction_median": float(
                (lh_f + hl_f + hh_f).median().item()
            ),
            "energy_high_freq_fraction_mean": float(
                (lh_f + hl_f + hh_f).mean().item()
            ),
        }
        # Niveau 2 : décompose LL à nouveau
        if LL.shape[-1] >= 2 and LL.shape[-2] >= 2:
            cAc2, cDc2 = _haar_1d_along_dim(LL, dim=-1)
            LL2, LH2 = _haar_1d_along_dim(cAc2, dim=-2)
            HL2, HH2 = _haar_1d_along_dim(cDc2, dim=-2)
            e_LL2 = _energy(LL2)
            e_total2 = (e_LL2 + _energy(LH2) + _energy(HL2) + _energy(HH2)).clamp_min(1e-30)
            ll2_f = (e_LL2 / e_total2).float().flatten()
            results["energy_LL2_fraction_median"] = float(ll2_f.median().item())

        results["n_matrices"] = int(B * H)
        return results

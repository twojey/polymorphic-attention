"""
w1_pattern_complexity.py — Property W1 : complexité du pattern binaire de support.

Spec : DOC/CATALOGUE §W1 (proxy NIP — non-independence property).

V1 simplifié : on seuile A à τ × max|A|, on obtient un masque binaire,
et on calcule :
- entropie du nombre de paires de lignes "compatible-incompatible"
  (proxy NIP)
- VC-like dimension : nb max de lignes indépendantes parmi B[i, :]
  (toutes distinctes)

Plus la matrice code une structure "logiquement complexe" (NIP-failing),
plus l'entropie est élevée.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class W1PatternComplexity(Property):
    """W1 — entropie du pattern binaire après seuil."""

    name = "W1_pattern_complexity"
    family = "W"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, threshold_ratio: float = 0.10, eps_floor: float = 1e-30) -> None:
        if not 0.0 < threshold_ratio < 1.0:
            raise ValueError("threshold_ratio ∈ (0, 1)")
        self.threshold = threshold_ratio
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype).abs()
        max_per = A_work.flatten(start_dim=-2).max(dim=-1).values.unsqueeze(-1).unsqueeze(-1).clamp_min(self.eps_floor)
        bin_mask = (A_work > self.threshold * max_per).long()  # (B, H, N, N), 0/1

        # 1. Nombre de patterns de lignes distincts (proxy VC-like)
        # Convertir chaque ligne en hash via somme pondérée (rapide)
        bin_typed = bin_mask.to(A_work.dtype)
        # Cap N2 pour éviter overflow : si N2 > 60, hasher via somme · index
        if N2 > 60:
            indices = torch.arange(N2, device=A_work.device, dtype=A_work.dtype)
            row_hashes = bin_typed @ indices
        else:
            powers = torch.pow(
                torch.tensor(2.0, device=A_work.device, dtype=A_work.dtype),
                torch.arange(N2, device=A_work.device, dtype=A_work.dtype),
            )
            row_hashes = bin_typed @ powers  # (B, H, N)

        # Compte uniques par (B, H)
        # Vectorisation pure compliquée — fall-back loop B·H
        unique_counts: list[int] = []
        for b in range(B):
            for h in range(H):
                u = torch.unique(row_hashes[b, h])
                unique_counts.append(int(u.numel()))
        unique_t = torch.tensor(unique_counts, dtype=torch.float32)

        # 2. Entropie du pattern : pour chaque (B, H), prob de chaque pattern
        # et H(p). Approximation : sur les hashes, distribution empirique.
        entropies: list[float] = []
        for b in range(B):
            for h in range(H):
                hash_vals = row_hashes[b, h]
                _, counts = torch.unique(hash_vals, return_counts=True)
                p = counts.float() / counts.sum().clamp_min(self.eps_floor)
                p_safe = p.clamp_min(self.eps_floor)
                H_pat = -(p * p_safe.log()).sum().item()
                entropies.append(H_pat)
        H_t = torch.tensor(entropies, dtype=torch.float32)
        H_norm = H_t / max(math.log(N), self.eps_floor)  # normalisée par log(N)

        # 3. Density du masque
        density = bin_mask.float().mean(dim=(-1, -2)).flatten().float()

        return {
            "pattern_unique_rows_median": float(unique_t.median().item()),
            "pattern_unique_rows_max": float(unique_t.max().item()),
            "pattern_entropy_median": float(H_t.median().item()),
            "pattern_entropy_norm_median": float(H_norm.median().item()),
            "binary_mask_density_median": float(density.median().item()),
            "threshold_ratio": self.threshold,
            "n_matrices": int(B * H),
        }

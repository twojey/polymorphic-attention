"""
c4_renyi_entropy.py — Property C4 : entropie de Rényi par ligne.

Spec : DOC/00b §C4 "H_α = (1-α)⁻¹ log Σ p_i^α (α=2 typique)".

Famille paramétrée par α :
- α → 1 : Shannon (limite)
- α = 2 : collision entropy, H_2 = -log Σ p²
- α → ∞ : min-entropy, H_∞ = -log max p

H_2 est particulièrement informatif : Σ p² = "purity" (probabilité de
collision), liée à la concentration. exp(-H_2) = Σ p² ∈ [1/N, 1].
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class C4RenyiEntropy(Property):
    """C4 — entropie de Rényi par ligne pour plusieurs α (défaut: 2, ∞)."""

    name = "C4_renyi_entropy"
    family = "C"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        alphas: tuple[float, ...] = (2.0,),
        include_min_entropy: bool = True,
        eps_floor: float = 1e-30,
    ) -> None:
        for a in alphas:
            if a <= 0 or a == 1.0:
                raise ValueError(f"alpha={a} invalide (doit être > 0 et ≠ 1)")
        self.alphas = alphas
        self.include_min_entropy = include_min_entropy
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        row_sum = A_work.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        P = A_work / row_sum
        P_safe = P.clamp_min(self.eps_floor)
        log_N = math.log(N2)

        results: dict[str, float | int | str | bool] = {}
        for alpha in self.alphas:
            # H_α(p) = log(Σ p^α) / (1 − α)
            log_sum = P_safe.pow(alpha).sum(dim=-1).log()  # (B, H, N)
            H_alpha = log_sum / (1.0 - alpha)
            H_norm = H_alpha / log_N
            h_flat = H_alpha.float().flatten()
            n_flat = H_norm.float().flatten()
            tag = f"alpha_{alpha:.1f}".replace(".", "p")
            results[f"renyi_{tag}_median"] = float(h_flat.median().item())
            results[f"renyi_{tag}_mean"] = float(h_flat.mean().item())
            results[f"renyi_{tag}_norm_median"] = float(n_flat.median().item())

        if self.include_min_entropy:
            # H_∞(p) = -log max p
            min_ent = -P_safe.max(dim=-1).values.log()  # (B, H, N)
            min_ent_norm = min_ent / log_N
            m_flat = min_ent.float().flatten()
            n_flat = min_ent_norm.float().flatten()
            results["min_entropy_median"] = float(m_flat.median().item())
            results["min_entropy_mean"] = float(m_flat.mean().item())
            results["min_entropy_norm_median"] = float(n_flat.median().item())

        results["n_rows"] = int(B * H * N)
        return results

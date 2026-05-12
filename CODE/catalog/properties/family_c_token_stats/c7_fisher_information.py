"""
c7_fisher_information.py — Property C7 : information de Fisher par ligne softmax.

Spec : DOC/CATALOGUE §C7.

Pour une distribution catégorielle p = softmax(s), la matrice de Fisher
F par rapport à s est :

    F = diag(p) − p · pᵀ

C'est une matrice (N, N) semi-définie positive de rang N − 1, avec
eigenvalues max{p_i} (approx) et 0 (vecteur 1). Elle quantifie la
"curvature de la sortie softmax" en chaque ligne.

V1 : on calcule F par ligne A[t,:], et on rapporte trace(F), λ_max(F),
et nullity (= 1 toujours pour le vecteur 1).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class C7FisherInformation(Property):
    """C7 — stats de la matrice de Fisher par ligne softmax."""

    name = "C7_fisher_information"
    family = "C"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-30) -> None:
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        row_sum = A_work.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        P = A_work / row_sum  # row-stochastique

        # F[t,:,:] = diag(p_t) - p_t p_t^T, shape (B, H, N, N2, N2)
        # On évite cet objet 5-d ; on calcule directement les invariants par ligne.
        # trace(F) = Σ p_i (1 - p_i) = 1 - Σ p_i² (purity complément)
        purity = (P ** 2).sum(dim=-1)  # (B, H, N)
        trace_F = 1.0 - purity  # (B, H, N)

        # λ_max(F) : pour F = diag(p) - p p^T, λ_max ≈ max p_i (proxy)
        # En exact : λ_max est solution de det(F − λI) = 0, plus complexe.
        # On utilise max p comme proxy (borne supérieure).
        max_p = P.max(dim=-1).values  # (B, H, N)

        # det'(F) = produit des eigenvalues non-nulles ≈ Π p_i (= "Fisher determinant proxy")
        # En log : log det' F ≈ Σ log p_i, normalisé par N
        P_safe = P.clamp_min(self.eps_floor)
        log_det_proxy = P_safe.log().sum(dim=-1) / N2  # (B, H, N), log-moyenne

        tr_flat = trace_F.float().flatten()
        max_flat = max_p.float().flatten()
        ld_flat = log_det_proxy.float().flatten()

        return {
            "fisher_trace_median": float(tr_flat.median().item()),
            "fisher_trace_mean": float(tr_flat.mean().item()),
            "fisher_max_p_median": float(max_flat.median().item()),
            "fisher_max_p_mean": float(max_flat.mean().item()),
            "fisher_log_det_proxy_median": float(ld_flat.median().item()),
            "fraction_high_trace_above_0p9": float(
                (tr_flat > 0.9).float().mean().item()
            ),
            "n_rows": int(tr_flat.numel()),
        }

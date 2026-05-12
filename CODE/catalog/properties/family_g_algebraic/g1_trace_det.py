"""
g1_trace_det.py — Property G1 : trace, déterminant, eigenvalues stats.

Spec : DOC/00b §G1.

Invariants algébriques canoniques d'une matrice carrée :
- tr(A) = Σ A_ii, somme des eigenvalues
- det(A) = Π eigenvalues
- log|det(A)| (robuste numériquement à la magnitude)
- |eigenvalue|_max (rayon spectral)
- |eigenvalue|_min (sensibilité inversion)

Pour A softmax stochastique de chaque ligne sommant à 1, tr(A) ∈ [0, N]
et det(A) souvent très petit (rapport entre eigenvalues étalées).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class G1TraceDet(Property):
    """G1 — invariants tr, det, |eig|_max sur A (B, H, N, N)."""

    name = "G1_trace_det"
    family = "G"
    cost_class = 2  # eigvals = O(N³)
    requires_fp64 = True
    scope = "per_regime"

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            raise ValueError(f"A doit être carrée, reçu N={N} != {N2}")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)

        # Trace (vectorisé)
        trace = torch.diagonal(A_work, dim1=-2, dim2=-1).sum(dim=-1)  # (B, H)
        trace_flat = trace.float().flatten()

        # log|det| via slogdet (robuste)
        sign, logabsdet = torch.linalg.slogdet(A_work)
        # logabsdet : (B, H), peut être -inf si singulier
        logabsdet_flat = logabsdet.float().flatten()
        finite_mask = torch.isfinite(logabsdet_flat)
        finite_logabsdet = logabsdet_flat[finite_mask]

        # Eigenvalues (complexes en général pour non-symétrique)
        eigvals = torch.linalg.eigvals(A_work)  # (B, H, N) complex
        eigvals_abs = eigvals.abs()  # (B, H, N) float
        eig_max = eigvals_abs.max(dim=-1).values
        eig_min = eigvals_abs.min(dim=-1).values
        eig_max_flat = eig_max.float().flatten()
        eig_min_flat = eig_min.float().flatten()

        results: dict[str, float | int | str | bool] = {
            "trace_median": float(trace_flat.median().item()),
            "trace_mean": float(trace_flat.mean().item()),
            "trace_min": float(trace_flat.min().item()),
            "trace_max": float(trace_flat.max().item()),
            "logabsdet_finite_fraction": float(finite_mask.float().mean().item()),
            "eig_max_abs_median": float(eig_max_flat.median().item()),
            "eig_max_abs_mean": float(eig_max_flat.mean().item()),
            "eig_min_abs_median": float(eig_min_flat.median().item()),
            "spectral_radius_median": float(eig_max_flat.median().item()),
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }
        if finite_logabsdet.numel() > 0:
            results["logabsdet_median"] = float(finite_logabsdet.median().item())
            results["logabsdet_mean"] = float(finite_logabsdet.mean().item())
        else:
            results["logabsdet_median"] = float("nan")
            results["logabsdet_mean"] = float("nan")
        return results

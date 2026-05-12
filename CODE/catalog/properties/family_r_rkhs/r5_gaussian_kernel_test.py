"""
r5_gaussian_kernel_test.py — Property R5 : test "Gauss-like" du noyau.

Pour un noyau de Mercer K(x, y) = exp(-‖x − y‖² / σ²), les valeurs sur la
diagonale principale dominent et décroissent symétriquement. On teste si
A se comporte ainsi via :

1. Symétrie : ‖A − Aᵀ‖_F / ‖A‖_F (proche 0 = symétrique)
2. Décroissance diagonale : A[i, j] décroît-elle avec |i − j| ?
3. Diagonal-dominance : A[i, i] > A[i, j] pour j ≠ i

Score Gaussian-like = combinaison des trois. Permet de tester R3 indirect.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class R5GaussianKernelTest(Property):
    """R5 — score "Gauss-like" : symétrie + décroissance diagonale."""

    name = "R5_gaussian_kernel_test"
    family = "R"
    cost_class = 1
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
        if N != N2:
            return {"skip_reason": "non-square", "n_matrices": int(B * H)}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)

        # 1. Symétrie ratio
        sym_diff = (A_work - A_work.transpose(-1, -2)).flatten(start_dim=-2).norm(dim=-1)
        norm = A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)
        sym_ratio = sym_diff / norm  # (B, H), 0 = symétrique

        # 2. Décroissance diagonale (mean A[i, i±k] / A[i, i] pour k=1, 2, 4)
        diag = A_work.diagonal(dim1=-2, dim2=-1)  # (B, H, N)
        decay_results = {}
        for k in (1, 2, 4):
            if N <= k:
                continue
            offset_p = A_work.diagonal(offset=k, dim1=-2, dim2=-1)  # (B, H, N-k)
            offset_m = A_work.diagonal(offset=-k, dim1=-2, dim2=-1)
            # Compare to diag[k:] for offset_p, diag[:-k] for offset_m
            ratio_p = offset_p / diag[..., k:].clamp_min(self.eps_floor)
            ratio_m = offset_m / diag[..., :-k].clamp_min(self.eps_floor)
            decay_results[k] = ((ratio_p + ratio_m) / 2.0).float().median().item()

        # 3. Diagonal dominance
        row_max_off = A_work.clone()
        eye_mask = torch.eye(N, device=A_work.device, dtype=torch.bool)
        row_max_off[..., eye_mask] = float("-inf")
        max_off_diag = row_max_off.max(dim=-1).values  # (B, H, N)
        diag_dom = (diag > max_off_diag).float().mean(dim=-1)  # (B, H)

        sym_flat = sym_ratio.float().flatten()
        dom_flat = diag_dom.float().flatten()

        results = {
            "symmetry_ratio_median": float(sym_flat.median().item()),
            "diagonal_dominance_fraction_median": float(dom_flat.median().item()),
            "is_gauss_like_fraction": float(
                ((sym_flat < 0.1) & (dom_flat > 0.5)).float().mean().item()
            ),
            "n_matrices": int(B * H),
        }
        for k, v in decay_results.items():
            results[f"decay_offset_{k}_ratio_median"] = float(v)
        return results

"""
g4_charpoly_coefficients.py — Property G4 : coefficients du polynôme caractéristique.

Spec : DOC/CATALOGUE §G4.

Le polynôme caractéristique de A est p(λ) = det(λI − A) = Σ c_k λ^k.
Les coefficients c_k sont des invariants algébriques classiques :
- c_n = 1 (par construction)
- c_{n-1} = -tr(A)
- c_0 = (-1)^n det(A)

V1 : on calcule via les **Newton identities** liant c_k aux power sums
p_k = Σ λ_i^k = tr(A^k). C'est O(n³) par power computation.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class G4CharpolyCoefficients(Property):
    """G4 — coefficients du polynôme caractéristique via Newton identities."""

    name = "G4_charpoly_coefficients"
    family = "G"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, k_max: int = 4, eps_floor: float = 1e-30) -> None:
        if k_max < 1:
            raise ValueError("k_max ≥ 1")
        self.k_max = k_max
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            raise ValueError("A doit être carrée")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # power sums p_k = tr(A^k)
        Ak = torch.eye(N, device=A_work.device, dtype=A_work.dtype).expand_as(A_work).clone()
        power_sums: list[torch.Tensor] = []
        for k in range(1, self.k_max + 1):
            Ak = Ak @ A_work
            pk = torch.diagonal(Ak, dim1=-2, dim2=-1).sum(dim=-1)  # (B, H)
            power_sums.append(pk)

        # Newton's identities : c_{n-k} récurrence avec p_1..p_k
        # k · e_k = Σ_{i=1}^k (-1)^{i-1} e_{k-i} p_i, e_0 = 1
        # Les coefficients du polynôme caractéristique sont (-1)^k e_k
        e_prev: list[torch.Tensor] = [torch.ones_like(power_sums[0])]
        for k in range(1, self.k_max + 1):
            acc = torch.zeros_like(power_sums[0])
            for i in range(1, k + 1):
                sign = (-1.0) ** (i - 1)
                acc = acc + sign * e_prev[k - i] * power_sums[i - 1]
            ek = acc / k
            e_prev.append(ek)

        results: dict[str, float | int | str | bool] = {}
        for k in range(1, self.k_max + 1):
            ck_flat = e_prev[k].float().flatten()
            results[f"e_{k}_median"] = float(ck_flat.median().item())
            results[f"e_{k}_mean"] = float(ck_flat.mean().item())

        # Trace = p_1 = e_1 (déjà couvert), determinant ≈ (-1)^N · e_N (besoin k=N).
        # Pour k_max < N, on rapporte un proxy déterminant via produit power sums.
        results["k_max"] = self.k_max
        results["n_matrices"] = int(B * H)
        results["seq_len"] = int(N)
        return results

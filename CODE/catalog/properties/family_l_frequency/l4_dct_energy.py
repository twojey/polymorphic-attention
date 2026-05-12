"""
l4_dct_energy.py — Property L4 : concentration d'énergie DCT-II.

La DCT-II est plus naturelle que FFT pour signaux réels non-périodiques.
Pour A 2D, on applique DCT séparément sur lignes puis colonnes. Mesure
la fraction d'énergie capturée dans le coin top-k×k (basses fréquences).

Cohérent avec l'intuition : si A varie lentement (signature contraction),
la majorité de l'énergie est dans le coin DCT supérieur gauche.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _dct_matrix(N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """DCT-II orthogonal matrix (N, N)."""
    n = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # (1, N)
    k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)  # (N, 1)
    M = torch.cos(math.pi / N * (n + 0.5) * k)
    M[0, :] *= 1.0 / math.sqrt(2.0)
    M *= math.sqrt(2.0 / N)
    return M  # (N, N) orthogonal


@register_property
class L4DctEnergy(Property):
    """L4 — fraction d'énergie DCT dans coin top-k×k basses fréquences."""

    name = "L4_dct_energy"
    family = "L"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, k_fractions: tuple[float, ...] = (0.10, 0.25, 0.50)) -> None:
        for kf in k_fractions:
            if not 0.0 < kf <= 1.0:
                raise ValueError(f"k_fraction {kf} doit être ∈ (0, 1]")
        self.k_fractions = tuple(sorted(k_fractions))

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            return {"skip_reason": "non-square", "n_matrices": int(B * H)}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        M = _dct_matrix(N, device=A_work.device, dtype=A_work.dtype)
        # 2D DCT : C = M @ A @ M.T
        C = torch.einsum("ij,bhjk,kl->bhil", M, A_work, M.transpose(-1, -2))
        E_total = C.pow(2).sum(dim=(-2, -1)).clamp_min(1e-30)

        results: dict[str, float | int | str | bool] = {}
        for kf in self.k_fractions:
            k = max(1, int(round(N * kf)))
            E_low = C[..., :k, :k].pow(2).sum(dim=(-2, -1))
            ratio = (E_low / E_total).float().flatten()
            tag = f"{kf:.2f}".replace(".", "p")
            results[f"dct_energy_low_frac_{tag}_median"] = float(ratio.median().item())
            results[f"dct_energy_low_frac_{tag}_mean"] = float(ratio.mean().item())

        # Spectral entropy DCT
        p = C.pow(2).flatten(start_dim=-2) / E_total.unsqueeze(-1)
        p = p.clamp_min(1e-30)
        H_dct = -(p * p.log()).sum(dim=-1)  # nats
        H_dct_flat = H_dct.float().flatten()
        results["dct_entropy_nats_median"] = float(H_dct_flat.median().item())
        results["dct_entropy_normalized_median"] = float(
            (H_dct_flat / math.log(N * N)).median().item()
        )
        results["n_matrices"] = int(B * H)
        return results

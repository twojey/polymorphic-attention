"""
d3_subspace_angles.py — Property D3 : angles principaux entre row-spaces de têtes.

Spec : DOC/CATALOGUE §D3.

Pour chaque paire (h_1, h_2), calcule les valeurs singulières de
Q_1^T · Q_2 où Q_i est la base orthonormale du row-space de A[h_i]
(top-r SVD). Les arccos de ces σ sont les angles principaux entre
sous-espaces.

Métriques :
- angle_max_median : θ_max sur les paires (= "distance max" entre subspaces)
- angle_min_median : θ_1 (= alignement principal)
- fraction_aligned : θ_max < 30° → têtes capturent ~même information
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class D3SubspaceAngles(Property):
    """D3 — angles principaux entre row-spaces top-r des têtes."""

    name = "D3_subspace_angles"
    family = "D"
    cost_class = 3  # SVD × paires têtes
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, top_r: int = 4, eps_floor: float = 1e-12) -> None:
        if top_r < 1:
            raise ValueError(f"top_r doit être ≥ 1, reçu {top_r}")
        self.top_r = top_r
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if H < 2:
            return {
                "n_head_pairs": 0,
                "angle_max_median_deg": 0.0,
                "fraction_aligned": 1.0,
                "n_matrices": int(B * H),
            }

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        r = min(self.top_r, N, N2)

        # SVD batch
        U, S, Vh = torch.linalg.svd(A_work, full_matrices=False)
        # Row-space basis = top-r colonnes de V^T transpose = Vh[..., :r, :]^T
        # Mais on prend U pour col-space ou Vh pour row-space ? L'attention est
        # (B,H,N,N) avec lignes = positions query. Row-space des lignes = Vh^T.
        # Pour subspace angles, on utilise Vh[..., :r, :] comme bases (taille r × N).
        bases = Vh[..., :r, :]  # (B, H, r, N)

        # Pour chaque paire (h1, h2) : calculer sing values de bases[h1] @ bases[h2]^T
        triu = torch.triu_indices(H, H, offset=1)
        n_pairs = triu.shape[1]
        # Récupère les paires : (B, n_pairs, r, N) pour chaque côté
        B_h1 = bases[:, triu[0]]  # (B, n_pairs, r, N)
        B_h2 = bases[:, triu[1]]

        # Cross-product: (B, n_pairs, r, r)
        cross = torch.einsum("bpri,bpsi->bprs", B_h1, B_h2)
        cross_sigmas = torch.linalg.svdvals(cross)  # (B, n_pairs, r)
        # Clamp pour éviter acos(>1)
        cross_sigmas = cross_sigmas.clamp(-1.0 + self.eps_floor, 1.0 - self.eps_floor)
        angles = torch.acos(cross_sigmas)  # radians (B, n_pairs, r)

        # angle_min = arccos(largest sigma) = smallest angle (alignement)
        angle_min = angles[..., 0]  # (B, n_pairs)
        # angle_max = arccos(smallest sigma) = largest angle (separation)
        angle_max = angles[..., -1]

        amax_flat = angle_max.float().flatten()
        amin_flat = angle_min.float().flatten()

        rad2deg = 180.0 / math.pi
        return {
            "angle_max_median_deg": float(amax_flat.median().item() * rad2deg),
            "angle_max_mean_deg": float(amax_flat.mean().item() * rad2deg),
            "angle_min_median_deg": float(amin_flat.median().item() * rad2deg),
            "angle_min_mean_deg": float(amin_flat.mean().item() * rad2deg),
            "fraction_aligned": float(
                (amax_flat * rad2deg < 30.0).float().mean().item()
            ),
            "fraction_orthogonal": float(
                (amin_flat * rad2deg > 75.0).float().mean().item()
            ),
            "top_r": r,
            "n_head_pairs": int(n_pairs * B),
            "n_matrices": int(B * H),
        }

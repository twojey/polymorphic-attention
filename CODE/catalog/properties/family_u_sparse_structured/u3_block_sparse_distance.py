"""
u3_block_sparse_distance.py — Property U3 : distance à Block-sparse (B sparse + bloc).

Spec : DOC/CATALOGUE §U3 "Block-sparse".

V1 : on découpe A en grilles de blocs (b × b), on calcule la norme de
chaque bloc, et on garde les top-k blocs (par norme). On reporte ε_block_sparse
en fonction de k (sweep top-k = 5%, 10%, 25% des blocs).

Pour des attentions très focales (ω élevé), on attend top-5% des blocs
capturer >90% de l'énergie.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class U3BlockSparseDistance(Property):
    """U3 — distance à matrice bloc-sparse (top-k blocs gardés)."""

    name = "U3_block_sparse_distance"
    family = "U"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        block_size: int = 8,
        k_fractions: tuple[float, ...] = (0.05, 0.10, 0.25),
        eps_floor: float = 1e-30,
    ) -> None:
        if block_size < 1:
            raise ValueError(f"block_size doit être ≥ 1")
        for f in k_fractions:
            if not 0.0 < f <= 1.0:
                raise ValueError(f"k_fraction {f} doit être ∈ (0, 1]")
        self.block_size = block_size
        self.k_fractions = k_fractions
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        bs = self.block_size
        if N < bs or N2 < bs:
            return {"n_matrices": int(B * H), "skip_reason": "N too small for block_size"}

        # Pad si N non multiple de bs
        N_pad = ((N + bs - 1) // bs) * bs
        N2_pad = ((N2 + bs - 1) // bs) * bs

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        if N_pad != N or N2_pad != N2:
            padded = torch.zeros(B, H, N_pad, N2_pad, device=A_work.device, dtype=A_work.dtype)
            padded[..., :N, :N2] = A_work
            A_work = padded
            N, N2 = N_pad, N2_pad

        # Reshape en (B, H, m_rows, bs, m_cols, bs) puis (B, H, m_rows, m_cols, bs, bs)
        m_rows = N // bs
        m_cols = N2 // bs
        blocks = A_work.view(B, H, m_rows, bs, m_cols, bs).permute(0, 1, 2, 4, 3, 5)
        # Norme Frobenius par bloc
        block_norms = blocks.flatten(start_dim=-2).norm(dim=-1)  # (B, H, m_rows, m_cols)
        block_norms_flat = block_norms.flatten(start_dim=-2)  # (B, H, m_rows * m_cols)

        total_norm_sq = (block_norms_flat ** 2).sum(dim=-1)  # (B, H)
        total_norm = total_norm_sq.sqrt().clamp_min(self.eps_floor)

        results: dict[str, float | int | str | bool] = {}
        n_blocks_total = m_rows * m_cols
        for f in self.k_fractions:
            k = max(1, int(f * n_blocks_total))
            # Top-k blocs (par norme²)
            topk_norms_sq, _ = (block_norms_flat ** 2).topk(k=k, dim=-1)
            kept_energy_sq = topk_norms_sq.sum(dim=-1)
            # ε² = 1 − kept_energy² / total_energy² (energie résiduelle ratio)
            kept_ratio = (kept_energy_sq / total_norm_sq.clamp_min(self.eps_floor)).sqrt()
            eps_sq = 1.0 - kept_ratio
            eps = eps_sq.clamp_min(0.0).sqrt()  # ratio résiduel
            eps_flat = eps.float().flatten()
            tag = f"{f:.2f}".replace(".", "p")
            results[f"epsilon_block_sparse_top_{tag}_median"] = float(eps_flat.median().item())
            results[f"epsilon_block_sparse_top_{tag}_mean"] = float(eps_flat.mean().item())

        results["n_blocks_total"] = n_blocks_total
        results["block_size"] = bs
        results["n_matrices"] = int(B * H)
        return results

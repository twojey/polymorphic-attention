"""
o5_pan_recurrence.py — Property O5 : récurrence Pan (fast algorithms).

Spec : DOC/CATALOGUE §O5 "récurrence Pan-Schur / multiplications rapides
matrice à structure".

La théorie de Pan (Structured Matrices and Polynomials, 2001) montre
que pour une matrice à rang de déplacement r borné, on peut construire
un produit matrice-vecteur en O(rN log N) via des FFT successives. La
recurrence Pan-Schur fournit la factorisation explicite ; on l'estime
ici par un proxy.

V1 proxy : pour A donnée, on compare le coût de y = A·x via :
1. produit dense O(N²)
2. approximation FFT structurée O(N log N) — moyennée sur N tirages aléatoires
   de x ; erreur ‖y_exact − y_approx‖ / ‖y_exact‖

L'approximation est faite via projection FFT du produit matrice-vecteur :
    y_fft(k) = inverseFFT( fft(A_circ) ⊙ fft(x) ) — où A_circ est la
    circulante la plus proche de A (B1 baseline).

Si A est proche d'un produit circulant + low-rank → erreur faible.
Sinon → erreur élevée.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class O5PanRecurrence(Property):
    """O5 — proxy fast algorithm via approximation FFT-circulant + low-rank."""

    name = "O5_pan_recurrence"
    family = "O"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(
        self,
        n_test_vectors: int = 8,
        low_rank_correction_r: int = 2,
        eps_floor: float = 1e-30,
    ) -> None:
        self.n_test_vectors = n_test_vectors
        self.low_rank_r = low_rank_correction_r
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Circulante la plus proche de A : moyenne des diagonales = c
        # c[k] = mean over i of A[i, (i+k) mod N]
        # Pour simplifier, on extrait c en moyennant les diagonales
        c = torch.zeros(B, H, N, dtype=A_work.dtype, device=A_work.device)
        for k in range(N):
            # extract diagonal at offset k (with wrap-around)
            idx = (torch.arange(N) + k) % N
            diag_vals = A_work[..., torch.arange(N), idx]  # (B, H, N)
            c[..., k] = diag_vals.mean(dim=-1)

        # Low-rank correction : SVD du résidu E = A − circ(c)
        # Construire circ(c) explicitement
        rows = torch.arange(N).unsqueeze(0)
        cols = torch.arange(N).unsqueeze(1)
        circ_idx = (cols - rows) % N  # (N, N)
        C_circ = c[..., circ_idx]  # (B, H, N, N)
        E = A_work - C_circ
        U_E, S_E, Vh_E = torch.linalg.svd(E, full_matrices=False)
        r = min(self.low_rank_r, S_E.shape[-1])
        S_r = S_E[..., :r]
        U_r = U_E[..., :r]
        Vh_r = Vh_E[..., :r, :]
        E_approx = U_r @ torch.diag_embed(S_r) @ Vh_r
        A_approx = C_circ + E_approx

        # Erreurs sur n_test_vectors
        torch.manual_seed(0)
        errs_rel: list[float] = []
        for _ in range(self.n_test_vectors):
            x = torch.randn(B, H, N, 1, dtype=A_work.dtype, device=A_work.device)
            y_exact = A_work @ x
            y_approx = A_approx @ x
            err = (y_exact - y_approx).flatten(start_dim=-2).norm(dim=-1)
            ref = y_exact.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)
            rel = (err / ref).float().flatten()
            errs_rel.append(rel)
        errs_t = torch.cat(errs_rel)

        # Aussi rapport rang(E) / N
        rank_E = (S_E > 1e-10).float().sum(dim=-1).flatten()

        return {
            "pan_fft_lr_eps_median": float(errs_t.median().item()),
            "pan_fft_lr_eps_mean": float(errs_t.mean().item()),
            "pan_fft_lr_eps_p90": float(errs_t.quantile(0.90).item()),
            "fraction_eps_below_0p10": float((errs_t < 0.10).float().mean().item()),
            "rank_E_residual_median": float(rank_E.median().item()),
            "rank_E_residual_max": float(rank_E.max().item()),
            "low_rank_correction_r": int(r),
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }

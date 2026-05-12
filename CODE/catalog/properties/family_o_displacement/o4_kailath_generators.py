"""
o4_kailath_generators.py — Property O4 : générateurs de Kailath (G, B).

Spec : DOC/CATALOGUE §O4 "factorisation G · Bᵀ du déplacement".

Pour une matrice A et un opérateur de déplacement ∇(A) = A − Z·A·Zᵀ
(Sylvester Toeplitz), si rang(∇(A)) = r, on peut factoriser :
    ∇(A) = G · Bᵀ,   G ∈ ℝ^{N×r}, B ∈ ℝ^{N×r}
G et B sont appelés les **générateurs de Kailath**. Ils paramètrent
toute la structure Toeplitz-like de A en O(rN) au lieu de O(N²).

V1 mesure : on factorise ∇(A) via SVD (∇(A) = U diag(σ) Vᵀ), prend
les r premiers facteurs avec σ_i ≥ tol → G = U_r √diag(σ_r),
B = V_r √diag(σ_r). On retourne :
- ε_reconstruct : ‖∇(A) − G Bᵀ‖_F / ‖∇(A)‖_F pour r=2 (Toeplitz parfait)
- balance ratio : ‖G‖ / ‖B‖
- rank effectif (réutilise O1) — pour double-check
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _shift_down(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    Z = torch.zeros(n, n, device=device, dtype=dtype)
    if n > 1:
        idx = torch.arange(n - 1, device=device)
        Z[idx + 1, idx] = 1.0
    return Z


@register_property
class O4KailathGenerators(Property):
    """O4 — générateurs Kailath via factorisation rang-r de ∇(A)."""

    name = "O4_kailath_generators"
    family = "O"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, target_ranks: tuple[int, ...] = (1, 2, 4), eps_floor: float = 1e-30) -> None:
        self.target_ranks = target_ranks
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        Z = _shift_down(N, device=str(A_work.device), dtype=A_work.dtype)
        nabla = A_work - Z @ A_work @ Z.T  # (B, H, N, N)

        # SVD pleine (batched)
        U, S, Vh = torch.linalg.svd(nabla, full_matrices=False)  # U (B,H,N,N), S (B,H,N), Vh (B,H,N,N)

        nabla_norm = nabla.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)

        result: dict[str, float | int | str | bool] = {
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }

        for r in self.target_ranks:
            r = min(r, N)
            S_r = S[..., :r]  # (B, H, r)
            U_r = U[..., :, :r]
            Vh_r = Vh[..., :r, :]
            # ∇_r = U_r diag(S_r) Vh_r
            recon = U_r @ torch.diag_embed(S_r) @ Vh_r
            err = (nabla - recon).flatten(start_dim=-2).norm(dim=-1)  # (B, H)
            eps = (err / nabla_norm).float().flatten()
            # Générateurs : G = U_r diag(sqrt(S_r)), Bk = V_r diag(sqrt(S_r))
            sqrt_S = S_r.clamp_min(0.0).sqrt()
            G_norm = (U_r * sqrt_S.unsqueeze(-2)).flatten(start_dim=-2).norm(dim=-1)
            Bk_norm = (Vh_r.transpose(-2, -1) * sqrt_S.unsqueeze(-2)).flatten(start_dim=-2).norm(dim=-1)
            balance = (G_norm.float() / Bk_norm.float().clamp_min(self.eps_floor)).flatten()
            result[f"epsilon_r{r}_median"] = float(eps.median().item())
            result[f"epsilon_r{r}_mean"] = float(eps.mean().item())
            result[f"balance_r{r}_median"] = float(balance.median().item())

        # Test : minimum r pour epsilon < 0.01 (rang strict Kailath)
        eps_floor = 0.01
        min_r_per_mat: list[int] = []
        flat = S.reshape(B * H, N)
        nabla_norm_flat = nabla_norm.flatten()
        for i in range(B * H):
            sigmas_i = flat[i]
            cum_energy = (sigmas_i ** 2).cumsum(dim=-1).sqrt()
            total = nabla_norm_flat[i]
            err_per_r = (total ** 2 - cum_energy ** 2).clamp_min(0.0).sqrt()
            rel = err_per_r / total
            valid_r = (rel < eps_floor).nonzero(as_tuple=True)[0]
            if valid_r.numel() > 0:
                min_r_per_mat.append(int(valid_r[0].item()) + 1)
            else:
                min_r_per_mat.append(N)
        min_r_t = torch.tensor(min_r_per_mat, dtype=torch.float64)
        result["min_r_for_eps_0p01_median"] = float(min_r_t.median().item())
        result["min_r_for_eps_0p01_max"] = float(min_r_t.max().item())
        result["fraction_r_le_2"] = float((min_r_t <= 2).float().mean().item())

        return result

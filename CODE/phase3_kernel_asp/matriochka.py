"""
matriochka.py — Bases U/V Matriochka emboîtées hiérarchiquement.

Spec : DOC/03 §5.

U_max, V_max ∈ ℝ^{d × R_max} sont des paramètres statiques (pas de tirage
stochastique pendant l'entraînement). La correction low-rank au rang r est :

    ΔAttn_r(x) = U_max[:, :r] @ V_max[:, :r].T @ x

Init :
- Random : Xavier ou orthogonale par bloc
- Smart Init : top-K vecteurs singuliers des têtes spécialisées (phase 2.6),
  figés à l'init (pas de back-prop sur ces vecteurs)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MatriochkaInitConfig:
    strategy: str = "xavier"      # "xavier" | "orthogonal" | "smart"
    smart_init_vectors: torch.Tensor | None = None  # (d, K_total) si strategy == "smart"
    smart_init_freeze: bool = True


class MatriochkaBases(nn.Module):
    """U_max, V_max ∈ ℝ^{d × R_max} avec slicing au rang r."""

    def __init__(self, d_model: int, R_max: int, init: MatriochkaInitConfig | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.R_max = R_max
        init = init or MatriochkaInitConfig()
        self.U_max = nn.Parameter(torch.empty(d_model, R_max))
        self.V_max = nn.Parameter(torch.empty(d_model, R_max))
        self._init_bases(init)
        self._frozen_columns = 0
        if init.strategy == "smart" and init.smart_init_freeze and init.smart_init_vectors is not None:
            self._frozen_columns = init.smart_init_vectors.size(1)

    def _init_bases(self, init: MatriochkaInitConfig) -> None:
        if init.strategy == "xavier":
            nn.init.xavier_uniform_(self.U_max)
            nn.init.xavier_uniform_(self.V_max)
        elif init.strategy == "orthogonal":
            nn.init.orthogonal_(self.U_max)
            nn.init.orthogonal_(self.V_max)
        elif init.strategy == "smart":
            assert init.smart_init_vectors is not None, "smart strategy requires vectors"
            K = init.smart_init_vectors.size(1)
            assert K <= self.R_max, f"K={K} > R_max={self.R_max}"
            with torch.no_grad():
                self.U_max[:, :K] = init.smart_init_vectors
                self.V_max[:, :K] = init.smart_init_vectors  # symétrique en V1
                # le reste : random
                if K < self.R_max:
                    nn.init.xavier_uniform_(self.U_max[:, K:])
                    nn.init.xavier_uniform_(self.V_max[:, K:])
        else:
            raise ValueError(f"Stratégie d'init inconnue : {init.strategy}")

    def freeze_smart_columns(self) -> None:
        """Empêche le back-prop sur les K premières colonnes (smart init)."""
        if self._frozen_columns > 0:
            # Ces columns sont gérés par un hook sur le gradient
            def _hook(grad: torch.Tensor) -> torch.Tensor:
                grad = grad.clone()
                grad[:, : self._frozen_columns] = 0
                return grad

            self.U_max.register_hook(_hook)
            self.V_max.register_hook(_hook)

    def slice_at_rank(self, r: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retourne U[:, :r], V[:, :r]."""
        r = max(0, min(r, self.R_max))
        return self.U_max[:, :r], self.V_max[:, :r]

    def correction(self, x: torch.Tensor, r: int) -> torch.Tensor:
        """ΔAttn_r(x) = (U @ V^T) @ x. x : (B, N, d). Sortie : (B, N, d)."""
        U_r, V_r = self.slice_at_rank(r)
        if r == 0:
            return torch.zeros_like(x)
        # (d, r) @ (r, d) = (d, d) → broadcast sur (B, N, d)
        proj = U_r @ V_r.T
        return x @ proj.T

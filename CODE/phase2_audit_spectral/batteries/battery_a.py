"""
battery_a.py — Batterie A : fitting et identification de classe structurelle.

Spec : DOC/02 §5c, §2.6b ROADMAP.

Pour chaque matrice A :
- A.1 : ε_C = ‖A − project_C(A)‖_F / ‖A‖_F pour C ∈ {Toeplitz, Hankel, Cauchy, Vandermonde, Identity}
- A.2 : class_best = argmin_C ε_C
- A.3 : composition additive A ≈ M_1 + M_2
- A.4 : composition multiplicative A ≈ M_1 · M_2 (V2, plus complexe)

Projecteurs structurels :
- Toeplitz : moyenne par diagonale
- Hankel   : moyenne par anti-diagonale
- Identity : conserve la diagonale principale uniquement
- Cauchy   : projection low-rank dans l'espace des matrices Cauchy paramétrées (V2)
- Vandermonde : V2

V1 : Toeplitz, Hankel, Identity (les trois projecteurs analytiques fermés).
Cauchy/Vandermonde sont mises de côté pour V2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def project_toeplitz(A: torch.Tensor) -> torch.Tensor:
    """Projection orthogonale (Frobenius) sur l'espace des matrices Toeplitz.

    A : (..., M, N). T[i, j] = moyenne des A[i', j'] sur i' - j' = i - j.
    """
    *batch, M, N = A.shape
    out = torch.zeros_like(A)
    for d in range(-(M - 1), N):
        diag = torch.diagonal(A, offset=d, dim1=-2, dim2=-1)  # (..., k)
        if diag.numel() == 0:
            continue
        mean = diag.mean(dim=-1, keepdim=True)
        # broadcast mean back into the diagonal positions
        diag_size = diag.size(-1)
        i_start = max(0, -d)
        j_start = max(0, d)
        for k in range(diag_size):
            out[..., i_start + k, j_start + k] = mean.squeeze(-1)
    return out


def project_hankel(A: torch.Tensor) -> torch.Tensor:
    """Projection orthogonale sur les matrices de Hankel.

    H[i, j] dépend de i + j. Moyenne par anti-diagonale.
    """
    *batch, M, N = A.shape
    out = torch.zeros_like(A)
    for s in range(M + N - 1):
        # collecte antidiag i+j = s
        positions = [(i, s - i) for i in range(max(0, s - N + 1), min(M, s + 1))]
        if not positions:
            continue
        vals = torch.stack([A[..., i, j] for i, j in positions], dim=-1)
        mean = vals.mean(dim=-1, keepdim=True)
        for i, j in positions:
            out[..., i, j] = mean.squeeze(-1)
    return out


def project_identity(A: torch.Tensor) -> torch.Tensor:
    """Garde uniquement la diagonale principale."""
    *batch, M, N = A.shape
    out = torch.zeros_like(A)
    diag = torch.diagonal(A, dim1=-2, dim2=-1)  # (..., min(M,N))
    k = diag.size(-1)
    idx = torch.arange(k, device=A.device)
    # affecte sur la diagonale
    out[..., idx, idx] = diag
    return out


PROJECTORS_V1 = {
    "toeplitz": project_toeplitz,
    "hankel": project_hankel,
    "identity": project_identity,
}


@dataclass
class BatteryAResult:
    epsilon: dict[str, float] = field(default_factory=dict)         # par classe
    class_best: str = ""
    epsilon_best: float = float("inf")
    composition_additive: dict[str, float] | None = None             # A.3 : (class1, class2) -> ε résiduel


def fit_class(A: torch.Tensor, projector_name: str) -> float:
    """ε_C = ‖A − P_C(A)‖_F / ‖A‖_F."""
    proj = PROJECTORS_V1[projector_name](A)
    num = (A - proj).norm()
    den = A.norm().clamp_min(1e-30)
    return float((num / den).item())


def fit_class_with_residual(A: torch.Tensor, projector_name: str) -> tuple[float, torch.Tensor]:
    """Comme fit_class mais retourne aussi le résidu pour analyse aval."""
    proj = PROJECTORS_V1[projector_name](A)
    residual = A - proj
    eps = float((residual.norm() / A.norm().clamp_min(1e-30)).item())
    return eps, residual


def fit_classes_per_regime(
    A_per_regime: dict[tuple, torch.Tensor],
    *,
    classes: tuple[str, ...] = ("toeplitz", "hankel", "identity"),
) -> dict[tuple, BatteryAResult]:
    """Pour chaque régime, fit toutes les classes et identifie la dominante."""
    out: dict[tuple, BatteryAResult] = {}
    for regime_key, A in A_per_regime.items():
        result = BatteryAResult()
        for c in classes:
            result.epsilon[c] = fit_class(A, c)
        result.class_best = min(result.epsilon, key=result.epsilon.__getitem__)
        result.epsilon_best = result.epsilon[result.class_best]
        out[regime_key] = result
    return out


def fit_additive_composition(
    A: torch.Tensor,
    *,
    class1: str = "toeplitz",
    class2: str = "hankel",
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """A ≈ M1 + M2. Ajustement séquentiel : projeter A sur class1, soustraire,
    projeter le résidu sur class2. ε_residuel après deux projections.
    """
    M1 = PROJECTORS_V1[class1](A)
    residual1 = A - M1
    M2 = PROJECTORS_V1[class2](residual1)
    residual2 = residual1 - M2
    eps = float((residual2.norm() / A.norm().clamp_min(1e-30)).item())
    return eps, M1, M2

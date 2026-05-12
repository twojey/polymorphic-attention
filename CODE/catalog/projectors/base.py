"""
base.py — interface Projector pour les classes de matrices structurées.

Un Projector encapsule UNE classe (Toeplitz, Hankel, Cauchy, Vandermonde,
Butterfly, ...) avec son opération de projection orthogonale Frobenius.

Pattern : instance-based (et non classmethod) parce que certaines classes
(Cauchy, Vandermonde) ont des paramètres (poles, base) qui doivent vivre
dans l'instance. Pour les classes paramètre-free (Toeplitz, Hankel,
Identity), instancier `Toeplitz()` reste trivial.

Convention : `project(A)` accepte (..., M, N), retourne même shape. Vectorisé
sur les dimensions batch (B, H) et autres préfixes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Projector(ABC):
    """Une classe de matrices structurées avec sa projection orthogonale.

    Sous-classes :
    - Parameter-free : `Toeplitz()`, `Hankel()`, `Identity()`
    - Paramétrées : `Cauchy(poles_x, poles_y)`, `Vandermonde(base)`, ...

    Convention de naming : la sous-classe expose `name` (snake_case) +
    `family` (lettre DOC/00b, généralement "B" pour structurelles).
    """

    name: str = ""
    family: str = "B"  # défaut famille B structurelles

    @abstractmethod
    def project(self, A: torch.Tensor) -> torch.Tensor:
        """Projection orthogonale (Frobenius) de A sur la classe.

        A : (..., M, N), retourne (..., M, N) projeté.
        """
        ...

    def epsilon(self, A: torch.Tensor) -> torch.Tensor:
        """ε_C(A) = ‖A − P_C(A)‖_F / ‖A‖_F par batch elem.

        Retourne tensor (...,) où les dims (M, N) sont aplaties dans la norme.
        Valeur ∈ [0, 1] : 0 = A ∈ classe parfaite, 1 = orthogonale à la classe.
        """
        proj = self.project(A)
        num = (A - proj).flatten(start_dim=-2).norm(dim=-1)
        den = A.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)
        return num / den

    def residual(self, A: torch.Tensor) -> torch.Tensor:
        """A − P_C(A). Utilisé par Batterie B pour analyse résidu."""
        return A - self.project(A)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"

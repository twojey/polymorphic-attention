"""Projectors — primitives mathématiques structures (Toeplitz, Hankel, Cauchy, ...).

Spec : DOC/CONTEXT.md §Projector. Un Projector implémente UNE classe de
matrices structurées via :
- `project(A) → A_projected` : projection orthogonale Frobenius
- `epsilon(A) → tensor scalaire` : ε_C = ‖A − P_C(A)‖_F / ‖A‖_F
- `residual(A) → A − P_C(A)` : pour analyse Batterie B

Consommé par les Properties de family B (structurelles) du catalogue DOC/00b.
"""

from catalog.projectors.base import Projector
from catalog.projectors.identity import Identity
from catalog.projectors.toeplitz import Toeplitz
from catalog.projectors.hankel import Hankel

__all__ = [
    "Projector",
    "Identity",
    "Toeplitz",
    "Hankel",
]

"""Projectors — primitives mathématiques structures (Toeplitz, Hankel, Cauchy, ...).

Spec : DOC/00_FONDATIONS.md §Projector. Un Projector implémente UNE classe de
matrices structurées via :
- `project(A) → A_projected` : projection orthogonale Frobenius
- `epsilon(A) → tensor scalaire` : ε_C = ‖A − P_C(A)‖_F / ‖A‖_F
- `residual(A) → A − P_C(A)` : pour analyse Batterie B

Consommé par les Properties de family B (structurelles) du catalogue DOC/CATALOGUE.
"""

from catalog.projectors.base import Projector
from catalog.projectors.banded import Banded
from catalog.projectors.block_diagonal import BlockDiagonal
from catalog.projectors.butterfly_mask import ButterflyMask
from catalog.projectors.cauchy import Cauchy
from catalog.projectors.hankel import Hankel
from catalog.projectors.identity import Identity
from catalog.projectors.monarch_mask import MonarchMask
from catalog.projectors.toeplitz import Toeplitz

__all__ = [
    "Projector",
    "Banded",
    "BlockDiagonal",
    "ButterflyMask",
    "Cauchy",
    "Hankel",
    "Identity",
    "MonarchMask",
    "Toeplitz",
]

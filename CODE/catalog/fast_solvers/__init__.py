"""
fast_solvers — algorithmes rapides pour matrices structurées (Kailath/Pan).

Spec : DOC/CATALOGUE §1.2 "Cadre théorique : rang de déplacement".

Ces solveurs servent de **référence empirique** pour valider la
prédiction faite par les Properties O1/O4 (rang de déplacement Toeplitz)
et O2 (Cauchy) : si une matrice A a rang de déplacement borné, on doit
pouvoir lui appliquer ces solveurs avec une erreur faible.

Algorithmes implémentés :
- `levinson_durbin_solve(T_row, T_col, b)` : O(N²) Toeplitz système A·x = b
- `cauchy_solve(x, y, b, ...)` : Cauchy matrix-vector product O(N²) via
  GKO factorisation, O(N log² N) via Trummer en V2.
- `displacement_apply(G, B, x, op_type)` : applique une matrice
  structurée définie par ses générateurs (G, B) à un vecteur.

Module porté volontairement à API simple : `solver(A, x) → A·x` + erreur
relative reportée. Le but n'est pas la performance maximale (qui exige
FFT structurées O(N log² N) non encore impl) mais la **fidélité
mathématique** comme oracle de validation pour Sprint C.
"""

from catalog.fast_solvers.levinson import (
    levinson_durbin_solve,
    toeplitz_from_first_row_col,
    toeplitz_matvec,
)
from catalog.fast_solvers.cauchy import (
    cauchy_matrix,
    cauchy_matvec_naive,
    cauchy_solve,
)
from catalog.fast_solvers.displacement import (
    displacement_residual_norm,
    extract_displacement_generators,
    sylvester_displacement,
)

__all__ = [
    "levinson_durbin_solve",
    "toeplitz_from_first_row_col",
    "toeplitz_matvec",
    "cauchy_matrix",
    "cauchy_matvec_naive",
    "cauchy_solve",
    "displacement_residual_norm",
    "extract_displacement_generators",
    "sylvester_displacement",
]

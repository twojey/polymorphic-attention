"""Oracles — adapters fournissant des matrices d'attention à la Battery.

Spec : DOC/CONTEXT.md §Oracle.

Pattern : interface AbstractOracle + adapters concrets par domaine
(SMNIST, LL, vision, code). Permet la comparaison cross-domain qui est le
cœur de la Partie 1 (livrable scientifique "Mathematical Signatures of
Attention").
"""

from catalog.oracles.base import AbstractOracle, AttentionDump, RegimeSpec
from catalog.oracles.synthetic import SyntheticOracle

__all__ = [
    "AbstractOracle",
    "AttentionDump",
    "RegimeSpec",
    "SyntheticOracle",
]

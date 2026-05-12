"""Oracles — adapters fournissant des matrices d'attention à la Battery.

Spec : DOC/00_FONDATIONS.md §Oracle.

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


def __getattr__(name: str):
    """Lazy imports : SMNISTOracle (dépend phase1) + LLOracle (Sprint S7)."""
    if name == "SMNISTOracle":
        from catalog.oracles.smnist import SMNISTOracle
        return SMNISTOracle
    if name == "LLOracle":
        from catalog.oracles.language import LLOracle
        return LLOracle
    if name == "LLModelSpec":
        from catalog.oracles.language import LLModelSpec
        return LLModelSpec
    raise AttributeError(f"module 'catalog.oracles' has no attribute {name!r}")

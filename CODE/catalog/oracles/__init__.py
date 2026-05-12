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
    """Lazy imports : SMNIST, LL, Vision, Code Oracles."""
    if name == "SMNISTOracle":
        from catalog.oracles.smnist import SMNISTOracle
        return SMNISTOracle
    if name == "LLOracle":
        from catalog.oracles.language import LLOracle
        return LLOracle
    if name == "LLModelSpec":
        from catalog.oracles.language import LLModelSpec
        return LLModelSpec
    if name == "VisionOracle":
        from catalog.oracles.vision import VisionOracle
        return VisionOracle
    if name == "VisionModelSpec":
        from catalog.oracles.vision import VisionModelSpec
        return VisionModelSpec
    if name == "CodeOracle":
        from catalog.oracles.code import CodeOracle
        return CodeOracle
    if name == "CodeModelSpec":
        from catalog.oracles.code import CodeModelSpec
        return CodeModelSpec
    raise AttributeError(f"module 'catalog.oracles' has no attribute {name!r}")

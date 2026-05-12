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
    """Lazy imports : SMNIST, LL, Vision, Code Oracles + backends."""
    if name == "SMNISTOracle":
        from catalog.oracles.smnist import SMNISTOracle
        return SMNISTOracle
    if name in ("GPT2Oracle", "GPT2_VARIANTS"):
        import catalog.oracles.gpt2 as m
        return getattr(m, name)
    if name in ("LLOracle", "LLModelSpec", "MinimalLMBackend", "HFLanguageBackend"):
        import catalog.oracles.language as m
        return getattr(m, name)
    if name in ("VisionOracle", "VisionModelSpec", "MinimalViTBackend",
                "HFVisionBackend"):
        import catalog.oracles.vision as m
        return getattr(m, name)
    if name in ("CodeOracle", "CodeModelSpec", "MinimalCodeBackend"):
        import catalog.oracles.code as m
        return getattr(m, name)
    raise AttributeError(f"module 'catalog.oracles' has no attribute {name!r}")

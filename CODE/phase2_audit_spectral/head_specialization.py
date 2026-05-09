"""
head_specialization.py — Diagnostic de Spécialisation des têtes.

Spec : DOC/02 §5b.

Pour chaque tête h : spec_h = var(r_eff_h) à travers les régimes.
Têtes spécialisées = haut spec_h (varient fortement avec stress).
Têtes endormies = spec_h ≈ 0 et r_eff_h faible partout.

Statut : obligatoire si Phase 3 utilise Smart Init Matriochka, optionnel sinon.
PAS de pruning (DOC/02 §5b).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HeadDiagnostic:
    layer: int
    head: int
    mean_r_eff: float
    var_r_eff: float          # spec_h
    is_dormant: bool          # spec_h ≈ 0 et mean_r_eff faible


def diagnose_heads(
    *,
    r_eff: np.ndarray,         # (n_layers, n_heads, n_examples) ou (n_layers, n_heads, n_regimes)
    dormant_threshold: float = 0.5,
    spec_dormant_threshold: float = 0.1,
) -> list[HeadDiagnostic]:
    """Calcule spec_h par tête + détection dormance.

    Une tête est dormante si var(r_eff) < spec_dormant_threshold ET
    mean(r_eff) < dormant_threshold.
    """
    L, H, _ = r_eff.shape
    out: list[HeadDiagnostic] = []
    for ell in range(L):
        for h in range(H):
            vals = r_eff[ell, h]
            mean = float(vals.mean())
            var = float(vals.var())
            dormant = (var < spec_dormant_threshold) and (mean < dormant_threshold)
            out.append(HeadDiagnostic(layer=ell, head=h, mean_r_eff=mean, var_r_eff=var,
                                       is_dormant=dormant))
    return out


def top_specialized_heads(diagnostics: list[HeadDiagnostic], k: int = 8) -> list[HeadDiagnostic]:
    """Retourne les k têtes les plus spécialisées (var(r_eff) le plus haut)."""
    return sorted(diagnostics, key=lambda d: d.var_r_eff, reverse=True)[:k]

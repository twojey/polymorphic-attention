"""Batteries — orchestration Oracle × Properties → résultats.

Spec : DOC/CONTEXT.md §Battery.

Une Battery exécute une liste de Properties sur les régimes fournis par un
Oracle, agrège les sorties cross-régime en RegimeStats (V3.5 distributional),
et orchestre le logging MLflow.

Niveaux à venir : level_minimal, level_principal, level_extended, level_full,
level_research (cf. ROADMAP §3.9).
"""

from catalog.batteries.base import Battery, BatteryResults

__all__ = ["Battery", "BatteryResults"]

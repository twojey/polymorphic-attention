"""Sprints — orchestrateurs Sprint au-dessus des phases scientifiques.

Spec : DOC/ROADMAP §3.9 + DOC/CATALOGUE §Sprints.

Un **Sprint** est un objectif scientifique orienté résultat ("re-extraire
les dumps phase 1 V2", "lancer la batterie research sur les 9 dumps",
"phase 3 V3+ avec Backbone informé"). Il orchestre 1 ou plusieurs
phases existantes + un livrable.

Distinction Phase vs Sprint :
- **Phase** = unité algorithmique (extraction Oracle, audit SVD, training ASP)
  → CODE/phase{N}_*/run.py
- **Sprint** = objectif scientifique avec critères go/no-go
  → CODE/sprints/sprint_{lettre}_{nom}.py

Chaque SprintRunner expose `run()`, `checkpoint()`/`resume()` pour
interruption, et reporte un livrable structuré
(DOC/reports/sprints/sprint_X.md).
"""

from sprints.base import SprintBase, SprintResult, SprintStatus

__all__ = ["SprintBase", "SprintResult", "SprintStatus"]

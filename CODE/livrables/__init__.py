"""Livrables — génération artefacts scientifiques finaux du projet.

Spec : DOC/CATALOGUE §5 + DOC/ROADMAP §livrables.

Deux types de livrables :

**Partie 1 — Mathematical Signatures of Attention** (publication
indépendante) :
- `partie1_signatures.py` : table cross-Oracle Property × Oracle
- `partie1_predictions_vs_measured.py` : confrontation paris a priori vs mesures
- `cross_oracle_synthesis.py` : signatures par Oracle (résumé textuel)

**Partie 2 — ASP Verdict** (conclusion projet) :
- `partie2_asp_verdict.py` : GO/NO-GO ASP avec critères phase 5
- `partie2_pareto_curves.py` : courbes Pareto qualité vs budget

**Transverse** :
- `paper_figures.py` : génération figures matplotlib pour papers
- `latex_export.py` : export tables LaTeX
"""

from livrables.cross_oracle_synthesis import build_signatures_table
from livrables.paper_figures import generate_figure_signatures

__all__ = [
    "build_signatures_table",
    "generate_figure_signatures",
]

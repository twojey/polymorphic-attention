# DOC/paper/ — Structure des publications scientifiques

Deux livrables papier indépendants :

## Partie 1 — Mathematical Signatures of Attention Across Domains

Publication **inconditionnelle** : la classification est livrée dès que Sprint C + S4-S7 sont terminés, indépendamment du verdict ASP.

Voir `partie1/` :
- `outline.md` — plan détaillé
- `abstract.md` — résumé
- `predictions_a_priori.yaml` — paris pré-enregistrés (DOC/CATALOGUE §4.2)
- `figures/` — figures matplotlib générées par `livrables.paper_figures`

## Partie 2 — ASP Verdict : Polymorphic Attention Through Signal-Guided Routing

Publication **conditionnée à la réussite (ou échec instructif)** des Sprints D-G.

Voir `partie2/` :
- `outline.md` — plan détaillé
- `abstract.md` — résumé
- `pareto_analysis.md` — analyse coût/qualité
- `figures/`

## Génération automatique

```bash
# Construire la table cross-Oracle (Partie 1)
PYTHONPATH=CODE uv run python -m livrables.cross_oracle_synthesis \
    --results OPS/logs/sprint_C/results.json:OR \
              OPS/logs/sprint_S5/results.json:DV \
              OPS/logs/sprint_S7/results.json:LL \
    --output DOC/paper/partie1/

# Confronter paris a priori vs mesures (Partie 1)
PYTHONPATH=CODE uv run python -m livrables.partie1_predictions_vs_measured \
    --predictions DOC/paper/partie1/predictions_a_priori.yaml \
    --results OPS/logs/sprint_C/results.json:OR ... \
    --output DOC/paper/partie1/

# Verdict ASP (Partie 2)
PYTHONPATH=CODE uv run python -m livrables.partie2_asp_verdict \
    --test-results OPS/logs/sprint_G/test_results.json \
    --output DOC/paper/partie2/

# Figures (toutes)
PYTHONPATH=CODE uv run python -m livrables.paper_figures \
    --mode signatures --input DOC/paper/partie1/signatures_table.json \
    --output DOC/paper/partie1/figures/signatures.pdf
```

## Bibliography

`bibliography.bib` (BibTeX) — à compléter avec :
- Kailath & Sayed (1999) Fast Reliable Algorithms for Matrices with Structure
- Pan (2001) Structured Matrices and Polynomials
- Hackbusch (2015) Hierarchical Matrices
- Bochner (1933) Monotone Funktionen
- Vaswani et al. (2017) Attention Is All You Need
- Beltagy et al. (2020) Longformer
- Tay et al. (2022) Efficient Transformers: A Survey
- Choromanski et al. (2020) Rethinking Attention with Performers
- Wang et al. (2020) Linformer
- Katharopoulos et al. (2020) Linear Transformers
- Daras et al. (2024) Butterfly Transformer
- Dao et al. (2022) Monarch

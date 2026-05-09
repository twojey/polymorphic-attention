# DOC/reports — Rapports de phase

Un rapport par phase exécutée. Format Markdown, généré semi-automatiquement
par les drivers Hydra (`phase{N}_metrologie.run`, `phase{N}.report`).

## Convention de nommage

- `phase{N}_{run_id}.md` — rapport autogéré par le driver
- `phase{N}_{sprint}_summary.md` — synthèse manuelle de fin de sprint

## Workflow

1. **Pré-run** : rien à faire — le driver consomme la config Hydra commitée.
2. **Run** : driver produit `phase{N}_{run_id}.md` automatiquement, log les
   figures comme MLflow artifacts.
3. **Post-run** : revue manuelle, ajout de commentaires "lessons learned" en
   bas du rapport. Commit du rapport final.
4. **Fin de sprint** : compiler les rapports en `phase{N}_{sprint}_summary.md`
   avec la décision go/no-go et les recommandations pour le sprint suivant.

## Templates

- `phase1_template.md` — métrologie RCP via SSG
- `phase1b_template.md` — Identification Gate (3 signaux)
- `phase2_template.md` — Audit Spectral, SCH
- `phase3_template.md` — ASPLayer (post-phase-2 dérivation Backbone)
- `phase4_template.md` — Spectromètre + Curriculum
- `phase5_template.md` — Stress Test final + Pareto

Les templates sont des squelettes ; le driver Hydra remplit les sections.

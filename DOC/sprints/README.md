# Sprints — orchestrateurs au-dessus des phases

Spec : DOC/CATALOGUE.md §Sprints + DOC/ROADMAP §3.9.

Un **Sprint** est un objectif scientifique orienté résultat. Il orchestre une ou plusieurs phases existantes + un livrable Markdown.

## Liste des Sprints

| Sprint | Objectif | Pré-requis | Compute | Wall-clock |
|---|---|---|---|---|
| **B** | Re-extraction dumps phase 1 V2 | Oracle SMNIST entraîné | ~$0.30 pod CPU | 30 min - 1 h |
| **C** | Battery level_research × dumps | Sprint B done | ~$5-10 | 1 sem |
| **D** | Phase 3 V3+ Backbone informé | Sprint C report | ~$5-10 pod GPU | 1-2 sem |
| **E** | Phase 4a warm-up Spectromètre | Sprint D ckpt | ~$5-10 pod GPU | 1 jour |
| **F** | Phase 4b autonomous routing | Sprint E ckpt | ~$10-20 pod GPU | 2 jours |
| **G** | Phase 5 validation 5a-6c | Sprint F ckpt | ~$15-25 pod GPU | 3 jours |
| **S4** | SMNIST seq_len étendu 1024-4096 | Oracle SMNIST | ~$5 pod | 2 jours |
| **S5** | Vision Oracle (DINOv2) | transformers + HF auth | ~$10-20 pod | 1-2 jours |
| **S6** | Code Oracle (StarCoder) | transformers | ~$10-20 pod | 1-2 jours |
| **S7** | LL Oracle (Llama-3.2-1B) | transformers + HF auth | ~$10-20 pod | 2 jours |

## Lancer un Sprint

```bash
PYTHONPATH=CODE uv run python -m sprints.run --sprint B \
    --output OPS/logs/sprint_B \
    --oracle-checkpoint OPS/checkpoints/oracle_smnist.pt

PYTHONPATH=CODE uv run python -m sprints.run --sprint C \
    --output OPS/logs/sprint_C \
    --dumps-dir OPS/logs/sprint_B/dumps
```

Chaque Sprint :
- Sauvegarde checkpoint atomique → reprise après crash
- Logue métriques MLflow (si MLFLOW_TRACKING_URI)
- Évalue critères go/no-go explicites
- Produit `summary.json` + `report.md` dans output_dir

## Templates rapports

Voir `DOC/reports/sprints/*_template.md`. Un rapport est rempli **après** chaque Sprint avec les résultats observés.

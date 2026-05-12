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

**Méthode recommandée (production pod) — via launch_sprint.sh :**

```bash
# Sprint B (re-extraction SMNIST dumps) — nohup + watch logs
bash OPS/setup/launch_sprint.sh B --nohup --watch \
    --device cpu \
    -- --oracle-checkpoint OPS/checkpoints/oracle_e2f0b5e.ckpt

# Sprint C (Battery research × dumps)
bash OPS/setup/launch_sprint.sh C --nohup \
    -- --dumps-dir OPS/logs/sprints/B_re_extract/dumps

# Sprint S5 (Vision DINOv2 via HF Hub)
bash OPS/setup/launch_sprint.sh S5 --nohup --device cuda \
    --mlflow-uri http://localhost:5000 \
    -- --use-hf-dinov2
```

`launch_sprint.sh` ajoute :
- Strict mode bash (set -euo pipefail + trap ERR)
- Log persistant horodaté `OPS/logs/sprints/sprint_<ID>_<UTC>.log`
- Env vars BLAS=1 (anti-deadlock), PYTHONUNBUFFERED, MLflow URI
- Mode `--nohup` pour détacher le process
- Mode `--watch` pour tail -f en simultané

**Méthode directe Python (debug / dev local) :**

```bash
PYTHONPATH=CODE uv run python -m sprints.run --sprint B \
    --output OPS/logs/sprints/B \
    --oracle-checkpoint OPS/checkpoints/oracle_e2f0b5e.ckpt
```

## Garanties de robustesse

Chaque Sprint produit :
- **Checkpoint atomique** : reprise après crash (fingerprint mismatch → RuntimeError explicite)
- **Log horodaté UTC** : `<output_dir>/sprint.log` (append mode)
- **Manifest reproductible** : git hash + dirty flag, torch version, python, cuda, dans `summary.json`
- **Retry transients** : `shared.retry.retry_call` sur extract_regime (3 tentatives, backoff exp + jitter)
- **MLflow opt-in** : si `--mlflow-uri` ou `$MLFLOW_TRACKING_URI`
- **Critères go/no-go** tracés : tous les `_check_go_nogo` logués INFO + dans summary
- **Pas d'erreur silencieuse** : tous les `except` → `logger.exception` (traceback complet)

## Génération livrables après Sprints

```bash
# Tous les livrables Partie 1 d'un coup (post-Sprint C + S4-S7)
PYTHONPATH=CODE uv run python -m livrables.run_all \
    --results OPS/logs/sprints/C_catalog_full/results.json:OR \
              OPS/logs/sprints/S5_vision/results.json:DV \
              OPS/logs/sprints/S7_ll/results.json:LL \
    --predictions DOC/paper/partie1/predictions_a_priori.yaml \
    --output DOC/paper/partie1/

# Verdict ASP final (post-Sprint G)
PYTHONPATH=CODE uv run python -m livrables.partie2_asp_verdict \
    --test-results OPS/logs/sprints/G_phase5_validation/test_results.json \
    --output DOC/paper/partie2/
```

## Templates rapports

Voir `DOC/reports/sprints/*_template.md`. Un rapport est rempli **après** chaque Sprint avec les résultats observés.

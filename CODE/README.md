# CODE — Code source ASP

Trois niveaux d'organisation :

1. **Modules transverses** : `shared/`, `infra/`
2. **Catalogue Partie 1** (prioritaire 2026-05-12) : `catalog/`
3. **Orchestration** : `sprints/`, `livrables/`
4. **Phases scientifiques** (Partie 2) : `phase{1, 1b, 2, 3, 4, 5}_*/`

## Arborescence

```
CODE/
├── shared/                          # Primitives réutilisables (checkpoint, retry, logging, mlflow)
├── infra/                           # MachineProfile (hardware abstraction)
│
├── catalog/                         # ⭐ Catalogue Partie 1 — 131 Properties / 23 familles
│   ├── properties/                  # 131 Properties classées A-W + N (catalogue complet)
│   ├── oracles/                     # 5 adapters : Synthetic, SMNIST, LL, Vision, Code
│   ├── batteries/                   # Battery + levels (minimal → research) + n_workers parallel
│   ├── projectors/                  # 8 projectors structurés
│   ├── fast_solvers/                # Levinson, Cauchy, Sylvester (oracles validation O/Q/U)
│   ├── run.py / report.py / cross_oracle.py / tests/
│
├── sprints/                         # ⭐ 10 orchestrateurs Sprint (B/C/D/E/F/G/S4-S7)
│   ├── base.py                      # SprintBase + manifest + checkpoint/resume + logging
│   ├── run.py                       # CLI : python -m sprints.run --sprint X
│   └── sprint_*.py
│
├── livrables/                       # ⭐ Génération artefacts paper Partie 1 + 2
│   ├── cross_oracle_synthesis.py    # Table Property × Oracle
│   ├── partie1_predictions_vs_measured.py
│   ├── partie1_signatures.py
│   ├── partie2_asp_verdict.py
│   ├── paper_figures.py             # Matplotlib heatmap/barplot/Pareto
│   └── run_all.py                   # Orchestrateur 1-shot tous livrables Partie 1
│
├── phase1_metrologie/               # Phase 1 — Oracle SMNIST + extraction
├── phase1b_calibration_signal/      # Phase 1.5 — Signaux S_KL/Grad/Spectral
├── phase2_audit_spectral/           # Phase 2 — SVD + SCH dictionnaire
├── phase3_kernel_asp/               # Phase 3 — ASPLayer entraînement
├── phase4_routage_budget/           # Phase 4 — Spectromètre + curriculum
└── phase5_pareto/                   # Phase 5 — tests validation 5a-6c
```

Chaque sous-dossier contient un README qui décrit l'attendu de la phase et les entrées/sorties prévues.

## Conventions

- **Stack** : PyTorch ≥ 2.11+cu128, Lightning Fabric, Hydra, uv. Voir `OPS/env/STACK.md`.
- **Tests** : `pytest CODE/`, **724 verts** + 1 skip OPENBLAS (catalog 131 + phase4/5 drivers)
- **Logging** : `shared.logging_helpers.setup_logging` + MLflow self-hosted optionnel
- **Checkpoint** : `shared.checkpoint.Checkpoint` atomic save+resume avec fingerprint
- **Retry** : `shared.retry.retry` / `retry_call` (backoff exp + jitter)
- **Format AttentionDump** : `(attn: list[Tensor (B, H, N, N)], omegas, deltas, entropies, tokens, query_pos, metadata)` cf. `catalog/oracles/base.py`
- **Reproductibilité** : seeds Hydra + manifest auto SprintBase (git hash + dirty, torch, python, cuda)

## Règle d'ordre

Phases scientifiques (1 → 5) : chaque phase dépend du verdict de la précédente (cf. `DOC/falsifiabilite.md`).

Sprints (B → G) : orchestrent les phases avec contraintes go/no-go. Squelettes `sprint_*.py` peuvent tourner avec retry/checkpoint sans pod (`level=minimal` ou backend random init) — utile pour smoke tests.

## Quick start

```bash
# Catalog sur Oracle synthétique (smoke test, 1.5s)
PYTHONPATH=CODE uv run python -m catalog.run \
    --oracle synthetic --level research --output /tmp/cat_smoke

# Sprint B sur pod CPU (30 min)
bash OPS/setup/launch_sprint.sh B --nohup --watch -- \
    --oracle-checkpoint OPS/checkpoints/oracle_e2f0b5e.ckpt

# Tous les livrables Partie 1 en une commande
PYTHONPATH=CODE uv run python -m livrables.run_all \
    --results <results.json>:<oracle_id> ... \
    --predictions DOC/paper/partie1/predictions_a_priori.yaml \
    --output DOC/paper/partie1/
```

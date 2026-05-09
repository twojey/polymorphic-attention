# OPS/configs — Configurations d'expérience

## Principe

**Une config = une expérience.** Une config contient tout ce qui est nécessaire pour rejouer un run sans toucher au code : seed, hyperparamètres, dataset, hardware ciblé, version de stack.

Format : YAML, composé via Hydra. Pas de défauts implicites — toute valeur qui change un résultat doit apparaître dans une config.

## Arborescence

```
OPS/configs/
├── README.md
├── manifest_template.yaml    # template de manifest produit à chaque run
├── phase1/                   # phase 1 — RCP via SSG + multi-Oracle
├── phase1b/                  # phase 1.5 — Identification Gate (3 signaux)
├── phase2/                   # phase 2 — Audit Spectral (SCH)
├── phase3/                   # phase 3 — ASPLayer
├── phase4/                   # phase 4 — Spectromètre + Curriculum
└── phase5/                   # phase 5 — Stress Test
```

Chaque sous-dossier `phaseN/` accueillera, au fil des sprints :

- `oracle_<domaine>.yaml` — config de l'Oracle (phase 1)
- `dataset_split_<domaine>.yaml` — stratégie de split tri-partition
- `thresholds.yaml` — seuils pré-enregistrés (cf. règle T.1)
- `signals.yaml` — config des 3 signaux candidats (phase 1.5)
- `audit.yaml` — paramètres SVD batchée, batterie de tests (phase 2)
- `asp_layer.yaml` — backbone dérivé, R_max, init Matriochka (phase 3)
- `spectrometer.yaml` — Curriculum, λ_budget, distillation 4a/4b (phase 4)
- `stress_test.yaml` — 5a–5e + 6c (phase 5)

La liste exacte est définie au moment de chaque sprint, pas d'avance.

## Composition Hydra

Pattern recommandé : un fichier de config par axe d'expérience, composés au lancement.

```yaml
# OPS/configs/phase1/oracle_smnist.yaml
defaults:
  - /thresholds@_global_: thresholds_phase1
  - _self_

oracle:
  arch:
    n_layers: 6
    n_heads: 8
    d_model: 256
    attention_type: dense    # pas Flash au moment de l'extraction
  training:
    batch_size: 64
    optimizer: adamw
    lr: 3e-4
    bf16: true
  ...
```

Les overrides ligne de commande sont autorisés mais traçés dans le manifest :

```bash
uv run python -m asp.phase1.train_oracle \
    --config-name=oracle_smnist \
    oracle.training.lr=1e-4 \
    seed=42
```

## Pré-enregistrement (règle T.1)

Toute config "registered" doit être commitée **avant** le run. Le run pointe vers le commit via le `git_short_hash` dans le run W&B et dans le manifest local. Une config modifiée après run est **copiée et renommée** (`oracle_smnist_v2.yaml`), jamais écrasée.

Tags W&B associés :
- `status:exploratory` → run libre, jamais cité dans un rapport de phase
- `status:registered`  → run reproductible à partir d'un commit, citable comme preuve
- `status:invalidated` → run pré-enregistré dont la config a été modifiée a posteriori → invalidé

## Manifest de run

À chaque run, un manifest YAML est produit et commité dans `OPS/logs/manifests/<run_id>.yaml` (cf. `manifest_template.yaml`). Champs obligatoires :

- `run_id` (≡ nom W&B)
- `git_commit` (hash long)
- `git_short_hash` (7 chars, présent dans le nom W&B)
- `seed`
- `phase`, `sprint`, `domain`
- `hardware` (gpu, vram, cuda, driver, pod_id)
- `started_at`, `finished_at`, `duration_s`
- `wandb_url`
- `data_hash` (hash des données utilisées, anti-fuite)
- `status` (`registered`, `exploratory`, `invalidated`)

Sans manifest, le résultat ne compte pas (cf. OPS/README.md).

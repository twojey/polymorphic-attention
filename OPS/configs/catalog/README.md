# OPS/configs/catalog/ — Configurations Hydra du module catalog

## Fichiers

- [`base.yaml`](base.yaml) — defaults communs (oracle, battery, mlflow, robustness)
- [`smnist_principal.yaml`](smnist_principal.yaml) — niveau `principal` sur SMNIST (Sprint A1)
- [`smnist_research.yaml`](smnist_research.yaml) — niveau `research` (~60 props) sur SMNIST
- [`cross_oracle.yaml`](cross_oracle.yaml) — harness cross-Oracle (SMNIST + Synthetic + ... LL/Vision quand prêts)

## Conventions

Les configs sont à composer via Hydra :

```bash
PYTHONPATH=CODE uv run python -m catalog.run \
    --oracle smnist --level research \
    --checkpoint OPS/checkpoints/oracle_e2f0b5e.ckpt \
    --output OPS/logs/catalog/smnist_research
```

Cross-oracle :

```bash
# Convertir cross_oracle.yaml → oracles-spec JSON
PYTHONPATH=CODE uv run python -m catalog.cross_oracle \
    --oracles-spec OPS/configs/catalog/cross_oracle_spec.json \
    --level principal \
    --output OPS/logs/catalog/cross_oracle
```

## Sprints

- **Sprint A1** : `smnist_principal.yaml`
- **Sprint A2-A3** : `smnist_research.yaml` (~60 properties)
- **Sprint S5-S7** : `cross_oracle.yaml` (avec Oracle LL + Vision activés)

# OPS/configs/sprints/ — Configs Hydra par Sprint

Une config Hydra par Sprint. Le runner `python -m sprints.run` accepte aussi des arguments CLI directs (cf. `--help`).

| Config | Sprint | Compute | Pré-requis |
|---|---|---|---|
| `sprint_b.yaml` | Re-extraction dumps SMNIST | $0.30 CPU pod | Oracle SMNIST ckpt |
| `sprint_c.yaml` | Battery research × dumps | $5-10 | Sprint B done |
| `sprint_d.yaml` | Phase 3 V3+ informé | $5-10 GPU | Sprint C report |
| `sprint_e.yaml` | Phase 4a warm-up | $5-10 GPU | Sprint D ckpt |
| `sprint_f.yaml` | Phase 4b autonomous | $10-20 GPU | Sprint E ckpt |
| `sprint_g.yaml` | Phase 5 validation | $15-25 GPU | Sprint F ckpt |
| `sprint_s4.yaml` | SMNIST seq_len étendu | $5 | Oracle SMNIST ckpt |
| `sprint_s5.yaml` | Vision Oracle (DINOv2) | $10-20 | transformers + HF auth |
| `sprint_s6.yaml` | Code Oracle (StarCoder) | $10-20 | transformers |
| `sprint_s7.yaml` | LL Oracle (Llama) | $10-20 | transformers + HF auth |

## Lancer un Sprint

Via CLI direct :
```bash
PYTHONPATH=CODE uv run python -m sprints.run --sprint B \
    --output OPS/logs/sprints/B \
    --oracle-checkpoint OPS/checkpoints/oracle_smnist.pt
```

Via Hydra (overrides config) :
```bash
PYTHONPATH=CODE uv run python -m sprints.run --sprint C \
    --output OPS/logs/sprints/C \
    --dumps-dir OPS/logs/sprints/B/dumps
```

## Override paramètres

Modifier directement le YAML, ou utiliser overrides Hydra-style si on intègre Hydra dans `sprints.run` (V2).

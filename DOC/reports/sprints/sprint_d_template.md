# Rapport Sprint D — Phase 3 V3+ avec Backbone informé

> Spec : `DOC/03_phase_kernel_asp.md` + `CODE/sprints/sprint_d_phase3_v3.py`.

## Métadonnées

- **Sprint ID** : `D_phase3_v3`
- **Date** : `<YYYY-MM-DD>`
- **Backbone class identifié Sprint C** : `<butterfly|monarch|banded|...>`
- **Pod** : `<RunPod instance>`
- **Durée training** : `<heures>`

## Configuration

- **N epochs** : `<int>`
- **Optimizer** : `<AdamW|Lion|...>`
- **Learning rate** : `<float>`
- **ASPLayer Smart Init** : `<smart_init recipe>`

## Résultats

| Métrique | Oracle baseline | ASP V3 | Ratio | Statut |
|---|---|---|---|---|
| val_acc | 0.645 | _ | _ | _ |
| val_loss | _ | _ | _ | _ |
| inference latency (ms) | _ | _ | _ | _ |
| param count | _ | _ | _ | _ |

## Critères go/no-go

| Critère | Attendu | Observé | Statut |
|---|---|---|---|
| val_acc ASP ≥ 0.90 × Oracle | ≥ 0.580 | _ | _ |
| backbone class reconnu non-dense | true | _ | _ |
| training converge sans NaN | true | _ | _ |

## Décisions

- ✅ / ❌ Continuer Sprint E

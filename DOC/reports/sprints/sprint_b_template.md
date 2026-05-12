# Rapport Sprint B — Re-extraction dumps phase 1 V2

> Spec : `DOC/01_phase_metrologie.md` + `CODE/sprints/sprint_b_re_extract.py`.
> Runner : `python -m sprints.run --sprint B --output <output_dir>`.

## Métadonnées

- **Sprint ID** : `B_re_extract`
- **Date d'exécution** : `<YYYY-MM-DD>`
- **Git hash** : `<short_hash>`
- **Oracle utilisé** : `<oracle_id>` (checkpoint `<path>`)
- **Pod** : `<RunPod instance>`
- **Durée wall-clock** : `<minutes>`
- **Coût compute** : `<$>`

## Configuration

- **Sweep ω** : `<liste>`
- **Sweep Δ** : `<liste>`
- **n_examples_per_regime** : `<int>`
- **Device** : `<cpu|cuda>`

## Résultats

### Dumps produits

| ω | Δ | n_examples | Taille fichier | Path |
|---|---|---|---|---|
| 0 | 16 | _ | _ | _ |
| 0 | 64 | _ | _ | _ |
| _ | _ | _ | _ | _ |

### Critères go/no-go

| Critère | Attendu | Observé | Statut |
|---|---|---|---|
| min_dumps_produced | ≥ 8 / 9 | _ | _ |
| dump_validate (no shape mismatch) | 0 erreur | _ | _ |

## Surprises / problèmes rencontrés

- ...

## Décisions

- ✅ / ❌ Continuer Sprint C
- Notes pour reproduction : ...

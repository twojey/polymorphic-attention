# Rapport Sprint C — Battery level_research sur dumps Sprint B

> Spec : `DOC/CATALOGUE.md` §Battery + `CODE/sprints/sprint_c_catalog_full.py`.

## Métadonnées

- **Sprint ID** : `C_catalog_full`
- **Date** : `<YYYY-MM-DD>`
- **Battery level** : `research` (98+ Properties)
- **N régimes** : `<int>` (depuis dumps Sprint B)
- **Pod** : `<instance / VPS>`
- **Durée** : `<heures>`

## Résultats globaux

- **Properties activées** : `<int>` / 98
- **Properties produisant une valeur** : `<int>` (`<%>`)
- **Properties skippées** : `<int>` (raisons : <résumé>)

## Familles dominantes

Top 5 familles avec variance cross-régime > 30 % :

| Famille | Property exemple | Variance cross-régime |
|---|---|---|
| _ | _ | _ |

## Classes structurelles identifiées

| Classe | Test | Évidence (Property + valeur) |
|---|---|---|
| Low-rank | A1 r_eff_median < N/4 | _ |
| Toeplitz-like | O1 fraction_rank_le_2 > 0.5 | _ |
| Sparse | B4 sparse_fraction > 0.5 | _ |
| Block-diag | B5 ε < 0.15 | _ |
| Butterfly | U1 ε < 0.20 | _ |
| Monarch | U2 ε < 0.20 | _ |
| Hierarchical | Q2 + Q4 nestedness preserved | _ |
| Mercer PSD | R1 fraction_psd > 0.5 | _ |
| Bochner stationaire | R3 fraction_pass_bochner > 0.5 | _ |

## Découvertes inattendues

- ...

## Verdict Partie 1 (intermédiaire)

**Classe(s) dominante(s)** identifiée(s) : `<liste>`.

**Implication Sprint D** : informer le Backbone ASPLayer avec projector `<class>` (CODE/catalog/projectors/<class>.py).

## Critères go/no-go

| Critère | Attendu | Observé | Statut |
|---|---|---|---|
| ≥ 50 % Properties produisent une valeur | true | _ | _ |
| Au moins 1 classe structurelle identifiée | true | _ | _ |

## Décisions

- ✅ / ❌ Continuer Sprint D avec backbone `<class>`
- Réserves : ...

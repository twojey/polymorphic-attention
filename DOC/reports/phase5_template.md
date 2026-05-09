# Rapport phase 5 — Stress Test + Pareto + OOD + R_max/2

> Spec : [DOC/05_phase_pareto.md](../05_phase_pareto.md).

## Métadonnées
- **Run ID, git, sprint, domaine(s)**
- **ASPLayer évaluée** : `<run_id phase 4 source>`

## Test 5a — Identifiabilité

### 5a.i — Anti-fraude
- R_target médian sur bruit blanc : `<v>` (seuil ≤ 1.0)
- Verdict : `PASS | FAIL`
- *(FAIL = arrêt total du protocole.)*

### 5a.ii — Activation différentielle (V3.5)
| Condition | R_target moyen |
|---|---|
| Bruit | _ |
| Null | _ |
| Trivial | _ |
| Structuré | _ |

- diff_score (sum KL pairwise) : `<v>` (seuil > seuil_actif)
- structured > others : `oui/non`
- Verdict : `PASS | FAIL`

## Test 5b — Élasticité

- Lag de Réaction (rise) : `<v>` (seuil < 8 tokens)
- Lag de Réaction (fall) : `<v>`
- Symétrique : `oui/non`
- Verdict : `PASS | FAIL`

## Test 5c — SE et HT

| Modèle | Accuracy | Avg Rank | Inference Time | SE | HT |
|---|---|---|---|---|---|
| ASP | _ | _ | _ | _ | _ |
| Transformer dense | _ | _ | _ | _ | _ |
| SSM pur | _ | _ | _ | _ | _ |

- Critère SE_ASP ≥ 2 × SE_Transformer : `PASS | FAIL`
- Critère HT_ASP ≥ HT_SSM : `PASS | FAIL`

## Test 5d — Pareto domain-aware

### Sequence-domain
| Modèle | Quality | FLOPs/token | Memory peak | Latency | Sur frontière ? |
|---|---|---|---|---|---|

- Test décisif vs MoD (montée graduée R_target sur récursion progressive) : `<oui/non>`

### Vision-domain (si évalué)
*Tableau analogue.*

### Code-domain (si évalué)
*Tableau analogue.*

## Test 5e — OOD croisé

| Train sur | Eval sur | R_target moyen train | R_target moyen eval | Ratio | Passe ? |
|---|---|---|---|---|---|
| récursion (ω varié, Δ min) | binding (Δ varié, ω min) | _ | _ | _ | _ |
| binding | récursion | _ | _ | _ | _ |
| mixte | algorithmiques tierces | _ | _ | _ | _ |

## Test 6c — R_max/2 (V3.5 — claim "ASP > Oracle")

- r_med Oracle : `<v>`
- R_max ASP utilisé : `<r_med / 2>`
- Quality ASP : `<v>`
- Quality Oracle : `<v>`
- Ratio : `<v>`
- Verdict : `strict (≥ 95%) | partial (∈ [80%, 95%]) | fail`

## Suite d'évaluation

- 3 seeds × N ∈ {2¹⁰, 2¹², 2¹⁴, 2¹⁶}
- Benchmarks : Structure-MNIST OOD, LRA, Needle-in-a-haystack, Induction heads, LM

## Ablations

| Variant | Quality | R_target moyen | Notes |
|---|---|---|---|
| ASP avec α=1 figé | _ | _ | _ |
| ASP avec α=0 figé | _ | _ | _ |
| ASP sans Curriculum | _ | _ | _ |
| ASP sans distillation 4a | _ | _ | _ |
| ASP sans L_consistency | _ | _ | _ |

## Verdict final (DOC/05 §5.10)

**Conjonction stricte 5a + 5b + 5c + 5d + 5e** : `PASS | FAIL`

- Si PASS complet : preuve du concept ASP, écriture article.
- Si succès partiel : rapport documenté, retour amont ou clôture.

Réponse à la question scientifique cadrée :
> *« Peut-on, avec les outils mathématiques actuels, synthétiser une attention
> linéaire ou superlinéaire sans trade-off majeur à partir d'oracles
> quadratiques ? »*

**Réponse** : `OUI | NON | PARTIEL`

## Lessons learned (manuel)

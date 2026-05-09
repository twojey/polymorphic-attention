# Rapport phase 2 — Audit Spectral

> Spec : [DOC/02_phase_audit_spectral.md](../02_phase_audit_spectral.md).
> Driver : `phase2_audit_spectral.run`.

## Métadonnées

- **Run ID** : `<run_id>`
- **Git hash** : `<short_hash>`
- **Sprint** : 2
- **Domaine(s)** : `<domaines, multi-Oracle si Sprint 4>`
- **Set utilisé** : `audit_svd` (DOC/01 §8.6)

## Stress-Rank Map (V3.5 distributionnelle)

### Monovariate

| Axe | Valeur | n | médiane r_eff | IQR | p10 | p90 | tail_99 |
|---|---|---|---|---|---|---|---|
| ω | _ | _ | _ | _ | _ | _ | _ |
| Δ | _ | _ | _ | _ | _ | _ | _ |
| ℋ | _ | _ | _ | _ | _ | _ | _ |

### 2D croisé (heatmap médiane + IQR superposé)

- `<run_id>_srm_omega_delta.png`
- `<run_id>_srm_omega_entropy.png` (si applicable)
- `<run_id>_srm_delta_entropy.png`

## Loi de transfert r_target = a · ω^α · Δ^β · g(ℋ)

| Coef | Valeur | _ |
|---|---|---|
| log a | _ |
| α (ω) | _ |
| β (Δ) | _ |
| γ (ℋ) | _ |
| R² | _ |

Comparaison cross-domain (multi-Oracle, Sprint 4) : `universal | partial | domain-specific`.

## Diagnostic spécialisation des têtes (DOC/02 §5b)

- Têtes dormantes : `<n>` / `<total>`
- Top-K spécialisées : `<liste (layer, head, var(r_eff))>`

## Batterie A — Fitting des classes

| Régime | ε_toeplitz | ε_hankel | ε_identity | ε_compose_t+h | classe dominante |
|---|---|---|---|---|---|

## Batterie B — Analyse résidu

| Régime | norm_residual | top-k SVD ratio | FFT ratio |
|---|---|---|---|

PCA cross-régimes : composantes principales ; classe émergente ?

## Batterie D — Out-of-catalogue

- Régimes orphelins : `<liste>` (ε_min > 0.30 même avec composition)
- Asymétrie eigen/SVD : `<table par régime>`
- Hypothèses sur classes manquantes : `<liste>`

## Dictionnaire SCH (sortie clé pour phase 3)

| Régime | r_eff médian | classe optimale | composition recommandée |
|---|---|---|---|

## R_max recommandé chiffré

`R_max = <valeur>` (raffiné depuis recommandation phase 1).

## Verdict go/no-go phase 2 (DOC/02 §2.8)

**Décision** : `GO | NO-GO`

- SCH corroborée : forte (IQR/médiane < 0.30) | faible | rejetée
- % régimes orphelins : `<x%>`

## Recommandation Backbone phase 3

Selon dictionnaire SCH :
- Classes dominantes : `<toeplitz | hankel | cauchy | combinaison>`
- Type d'opérateur recommandé : `<convolution causale | SSM générique | interpolation rationnelle | composite>`
- Configurable par domaine ? (si SCH domain-spécifique)

## Lessons learned (manuel)

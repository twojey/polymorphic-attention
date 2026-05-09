# Rapport phase 3 — ASPLayer

> Spec : [DOC/03_phase_kernel_asp.md](../03_phase_kernel_asp.md). Driver à
> écrire post-phase-2 (Backbone dérivé du dictionnaire SCH).

## Métadonnées
- **Run ID, git, sprint, domaine**

## Backbone dérivé (post-phase-2)
- Classe dominante consommée : `<toeplitz | hankel | cauchy | composite>`
- Implémentation : `<package.module.Class>`
- Hyperparamètres : `<...>`

## Init Matriochka
- Stratégie : `xavier | orthogonal | smart`
- Si smart : K_total = `<n>` vecteurs des têtes spécialisées phase 2.6
- Vérif freeze : columns 0..K-1 figées ?

## Sanity checks (DOC/03 §3.6)

| Check | Critère | Valeur | Statut |
|---|---|---|---|
| Saturation (r=R_max ≈ Oracle) | qualité ≥ Oracle - tol | _ | _ |
| Effondrement (r=0 ≡ Backbone) | diff < 1e-5 | _ | _ |
| Monotonie q(r) | croissante | _ | _ |
| Lissité \|q(r+1) − 2q(r) + q(r−1)\| | < seuil | _ | _ |

## Heatmap R_target par distillation
*(Heatmap de Rang sur (ω, Δ) → baseline pour phase 4)*

## Recommandations phase 4
- β Soft-Mask : `<valeur>`
- λ_M (loss matriochka) : `<valeur>`
- λ_C (loss consistency) : `<valeur>`
- w_r (poids par rang) : `<table>`
- Type LayerNorm : `<conditionnée par r_t ? oui/non>`

## Verdict go/no-go phase 3
- 4 sanity checks passés : oui/non
- Si NO-GO : action — révision Backbone, R_max, ou λ_M/λ_C

## Lessons learned (manuel)

# Rapport phase 1.5 — Identification Gate

> Spec : [DOC/01b_phase_calibration_signal.md](../01b_phase_calibration_signal.md).
> Driver : `phase1b_calibration_signal.run`.

## Métadonnées

- **Run ID** : `<run_id>`
- **Git hash** : `<short_hash>`
- **Sprint** : `<N>`
- **Domaine** : `smnist | ...`
- **Oracle utilisé** : `<oracle_id phase 1>`

## Configuration pré-enregistrée

- Baseline KL : seed=200, n_calibration_examples=1024 (DOC/01b §8 piège 1)
- S_Spectral : K=64, τ=1e-3 (DOC/01b §8 piège 2)
- Sous-échantillonnage : 1 token tous les 8 (DOC/01b §8 piège 3)
- Bootstrap : n=2000, α=0.05

## Matrice de corrélation Spearman 3 × 3 (avec IC95%)

| Signal | ρ_ω | IC95 ω | ρ_Δ | IC95 Δ | ρ_ℋ | IC95 ℋ |
|---|---|---|---|---|---|---|
| S_KL | _ | _ | _ | _ | _ | _ |
| S_Grad | _ | _ | _ | _ | _ | _ |
| S_Spectral | _ | _ | _ | _ | _ | _ |

## Verdict par signal (critères 0.70 / 0.20)

| Signal | max(\|ρ_ω\|, \|ρ_Δ\|) | \|ρ_ℋ\| | Passé ? | Axes couverts |
|---|---|---|---|---|
| S_KL | _ | _ | ✓/✗ | _ |
| S_Grad | _ | _ | ✓/✗ | _ |
| S_Spectral | _ | _ | ✓/✗ | _ |

## Distillabilité (sous-phase 1.5b, si S_Spectral retenu)

- ρ Spearman student/teacher : `<rho>`
- MSE relative : `<mse_rel>`
- Critère ρ > 0.85 ET MSE_rel < 0.5 : `PASS | FAIL`
- Fallback choisi (si FAIL) : `S_Spectral direct | simplification Backbone`

## Verdict go/no-go phase 1.5 (DOC/01b §7)

**Décision** : `GO | NO-GO`

**Signaux retenus pour phase 4** : `<liste>`

Si NO-GO : *« l'allocation dynamique de rang par token est une illusion
statistique »* → arrêt du protocole.

## Figures

- `<run_id>_correlation_matrix.png`
- `<run_id>_S_KL_vs_omega.png`, idem Δ, ℋ
- `<run_id>_S_Spectral_vs_omega.png`, idem
- `<run_id>_distill_pred_vs_teacher.png`

## Lessons learned (manuel)

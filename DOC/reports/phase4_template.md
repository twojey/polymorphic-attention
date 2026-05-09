# Rapport phase 4 — Spectromètre + Curriculum + Distillation

> Spec : [DOC/04_phase_routage_budget.md](../04_phase_routage_budget.md).
> Driver à écrire post-phase-1.5 (signaux retenus connus).

## Métadonnées
- **Run ID, git, sprint, domaine**
- **Signaux consommés (de phase 1.5)** : `<liste>`

## Phase 4a (warm-up + distillation V3.5)

- Cible : p75 du r_eff Oracle par régime
- Loss asymétrique : γ = `<valeur ∈ [0.1, 0.3]>`
- Curriculum stage atteint : `easy | intermediate | hard`
- Convergence L_distillation : `<courbe>`

## Transition 4a → 4b (critères pré-enregistrés)

| Critère | Valeur | Seuil | Statut |
|---|---|---|---|
| Loss change sur fenêtre 50 | _ | < 1e-3 | _ |
| ρ Spearman R_pred ↔ R_target théorique | _ | > 0.80 | _ |
| Variance R_pred (anti-plafond) | _ | > 0.5 | _ |

## Phase 4b (apprentissage autonome)

- Drift R_target sur N premiers steps : `<%>` (cible < 30%)

## Diagramme de Phase

| (ω, Δ, ℋ) | R_target moyen | R_target p75 |
|---|---|---|

Vérif croissance avec axes de stress : `croissant sur ω ✓ / Δ ✓ / ℋ ✓`

## Courbe Pareto λ_budget

- 7 valeurs testées : `[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]`
- Frontière non dominée : `<liste de (λ, qualité, rang_moyen)>`
- Portion strictement dominante vs Transformer : `<oui/non>`

## Verdict go/no-go phase 4

**Décision** : `GO | NO-GO`

- Courbe Pareto avec dominance : `<oui/non>`
- Diagramme de Phase croissant : `<oui/non>`
- R_target corrélé au stress local : `<oui/non>`

Si NO-GO : retour phase 1.5 (signaux insuffisants) ou phase 3 (architecture).

## Lessons learned (manuel)

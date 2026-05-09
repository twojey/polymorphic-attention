# Documentation — Attention Superlinéaire Polymorphe (ASP)

Spécification protocolaire d'une attention polymorphe : complexité O(N) par défaut, allocation locale de capacité quasi-O(N²) sous contrôle de rang adaptatif. La thèse n'est pas une optimisation d'ingénierie : c'est une compression structurelle motivée par la métrologie.

## Lecture

| # | Document | Phase | Objet |
|---|----------|-------|-------|
| 0 | [00_vision.md](00_vision.md) | — | Thèse, hypothèses, livrable |
| 1 | [01_phase_metrologie.md](01_phase_metrologie.md) | RCP / SSG | Métrologie via le Structural Stress Generator (ω, Δ, ℋ) |
| 1.5 | [01b_phase_calibration_signal.md](01b_phase_calibration_signal.md) | Identification Gate | Calibration du signal de stress (3 candidats, 2 critères, distillabilité) |
| 2 | [02_phase_audit_spectral.md](02_phase_audit_spectral.md) | Audit Spectral | Oracle + SVD + r_eff + Stress-Rank Map + loi de transfert (corroboration de la SCH) |
| 3 | [03_phase_kernel_asp.md](03_phase_kernel_asp.md) | ASPLayer | Backbone + correction Matriochka, Soft-Mask |
| 4 | [04_phase_routage_budget.md](04_phase_routage_budget.md) | Spectromètre | Curriculum de Stress, λ_budget, Diagramme de Phase |
| 5 | [05_phase_pareto.md](05_phase_pareto.md) | Validation et Falsification | 5 tests : Identifiabilité, Élasticité, SE/HT, Pareto, OOD |

Annexes : [glossaire.md](glossaire.md), [falsifiabilite.md](falsifiabilite.md).

## État

Spécification uniquement. Aucune implémentation. Le choix de la stack (PyTorch / JAX / autre) est délibérément différé : il sera arrêté à la sortie de la phase 1, en fonction de l'instrumentation requise par la métrologie.

## Principe directeur

> On ne devine pas la complexité, on la mesure. Chaque phase produit un livrable falsifiable et un critère go/no-go explicite. Si une phase échoue, le protocole s'arrête — pas de bricolage métrique post-hoc.

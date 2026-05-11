# Documentation — projet ASP

> **Cadrage projet (clarifié 2026-05-11) — Deux livrables distincts** :
>
> **Partie 1 — Science fondamentale** : étude spectrale exhaustive des propriétés mathématiques des attentions, livrable une **classification + batterie de tests** réutilisable par toute la communauté. Catalogue dans [00b_classification_proprietes.md](00b_classification_proprietes.md). Ne peut pas "fail" : chaque mesure (positive ou négative) enrichit la classification.
>
> **Partie 2 — Validation hypothèse polymorphique (ASP)** : appliquer Partie 1 pour vérifier si l'attention sub-quadratique via **allocation dynamique guidée par signal observable** est viable. Une approche parmi d'autres (kernel approx, sparse, low-rank, state-space). Conditionnée à la classification Partie 1.

## Lecture

### Fondations
| # | Document | Partie | Objet |
|---|----------|--------|-------|
| 0 | [00_vision.md](00_vision.md) | — | Thèse, hypothèses, livrable (centré Partie 2 / ASP) |
| 0b | **[00b_classification_proprietes.md](00b_classification_proprietes.md)** | **1** | **Catalogue mathématique exhaustif (~70 propriétés sur 18 catégories A-V)** |
| 0c | [00c_predictions_signatures.md](00c_predictions_signatures.md) | **1** | Paris a priori (pré-enregistrés) sur les signatures attendues de 8 Oracles classiques (DT/LA/PF/LF/MB/HY/BB/RF) — falsifiable contre la batterie 00b |

### Phases ASP (Partie 2)
| # | Document | Phase | Objet |
|---|----------|-------|-------|
| 1 | [01_phase_metrologie.md](01_phase_metrologie.md) | RCP / SSG | Métrologie via le Structural Stress Generator (ω, Δ, ℋ) |
| 1.5 | [01b_phase_calibration_signal.md](01b_phase_calibration_signal.md) | Identification Gate | Calibration du signal de stress (3 candidats, 2 critères, distillabilité) — sous-cas de la classification 00b |
| 2 | [02_phase_audit_spectral.md](02_phase_audit_spectral.md) | Audit Spectral | Oracle + SVD + r_eff + Stress-Rank Map + loi de transfert |
| 3 | [03_phase_kernel_asp.md](03_phase_kernel_asp.md) | ASPLayer | Backbone + correction Matriochka, Soft-Mask |
| 4 | [04_phase_routage_budget.md](04_phase_routage_budget.md) | Spectromètre | Curriculum de Stress, λ_budget, Diagramme de Phase |
| 5 | [05_phase_pareto.md](05_phase_pareto.md) | Validation et Falsification | 5 tests : Identifiabilité, Élasticité, SE/HT, Pareto, OOD |

Annexes : [glossaire.md](glossaire.md), [falsifiabilite.md](falsifiabilite.md), [carnet_de_bord.md](carnet_de_bord.md).

## État

Implémentation en cours. Phase 1 (Oracle SMNIST) terminée. Phase 1.5 en cours (Run 2/3/4 sur pod RunPod CPU 2026-05-11). Stack : PyTorch + Fabric + Hydra + uv. Cf. [carnet_de_bord.md](carnet_de_bord.md) pour l'avancement chronologique et [../ROADMAP.md](../ROADMAP.md) pour la planification.

## Principe directeur

> On ne devine pas la complexité, on la mesure. Chaque phase produit un livrable falsifiable et un critère go/no-go explicite. Si une phase échoue, le protocole s'arrête — pas de bricolage métrique post-hoc.

> **Pour la Partie 1** : exhaustivité **prime** sur efficacité. La valeur scientifique du livrable réside dans sa couverture des classes mathématiques connues. Si tous les tests échouent sur un Oracle, c'est une preuve forte (soit O(N²) intrinsèque, soit nouvelle classe à théoriser).
>
> **Pour la Partie 2 (ASP)** : un seul signal qui passe les seuils suffit (§4.3 phase 1.5). Économie de moyens.

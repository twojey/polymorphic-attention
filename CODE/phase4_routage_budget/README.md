# Phase 4 — Calibration du Spectromètre

Spec : [DOC/04_phase_routage_budget.md](../../DOC/04_phase_routage_budget.md).

## Modules attendus

- **spectrometer/** — mini-MLP ou conv 1D `(d → 1)` ou `(d → R_max)`. Reçoit s_t (signal de surprise + état récurrent court), émet α_t ou m_t directement.
- **surprise/** — extracteurs de signal de surprise (résidu prédictif Backbone, KL local, norme de gradient).
- **budget_loss/** — `L_sparsity` : forme canonique (1/T) Σ_t α_t + variantes (pondérée Matriochka, L1, quadratique).
- **curriculum/** — orchestration du Curriculum de Stress en 3 étages (warm-up bruit → injection légère → stress maximal).
- **distillation/** — loss auxiliaire ‖α_t · R_max − r_target‖² active en sous-phase **4a uniquement**, ancrée sur la loi de transfert phase 2.
- **transition_gate/** — vérification des trois critères de bascule 4a → 4b (convergence L_distillation, corrélation Spearman > 0,80, absence de plafond) + mesure de la dérive aux N premiers steps de 4b.
- **training/** — boucle d'entraînement en deux temps (4a avec distillation, 4b sans) avec balayage logarithmique de λ_budget, schedule de warmup λ_budget = 0 → λ_budget cible.
- **diagnostics/** — distribution de R_target, corrélation R_target ↔ stress local, génération du **Diagramme de Phase**.
- **report/** — courbe Pareto (qualité, complexité) paramétrée par λ_budget + Diagramme de Phase.

## Entrées

- ASPLayer de phase 3 (avec sanity checks passés).
- Loi de transfert r_target = f(ω, Δ, ℋ) (distillation initiale).
- Liste des valeurs de λ_budget à explorer (5–7 points, log-spaced, pré-enregistrés).
- Choix du proxy de surprise (instruction empirique).

## Sorties

- Spectromètre entraîné (en mode 4b autonome, sans distillation) pour chaque λ_budget exploré.
- Courbe Pareto λ_budget → (qualité, complexité), mesurée *après* 4b.
- Diagramme de Phase final.
- Évidence d'alignement R_target ↔ stress (préparation phase 5).
- Rapport de bascule 4a → 4b par valeur de λ_budget (dérive mesurée).

## Critère de fin

Go/no-go phase 4 documenté.

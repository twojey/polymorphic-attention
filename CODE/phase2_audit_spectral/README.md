# Phase 2 — Audit Spectral et SCH

Spec : [DOC/02_phase_audit_spectral.md](../../DOC/02_phase_audit_spectral.md).

## Modules attendus

- **svd/** — calcul de la SVD batché sur les matrices d'attention **extraites sur le set `audit_svd` uniquement** (pas `train_oracle`). Cast FP64 vérifié, pas d'agrégation pré-SVD. Extraction du spectre (σ_k).
- **r_eff/** — calcul du rang effectif r_eff(θ) pour θ = 0,95 et 0,99.
- **stress_rank_map/** — agrégation par (ω, Δ, ℋ), génération de la table et des cartes 2D (Radar RCP).
- **transfer_law/** — régression sur la Stress-Rank Map → loi de transfert r_target = f(ω, Δ, ℋ). Diagnostics (R², résidus).
- **head_diagnostic/** — Diagnostic de Spécialisation des têtes : `spec_h = var(r_eff_h)` à travers les régimes. Distribution par couche, liste ordonnée têtes spécialisées. **Obligatoire si phase 3 utilise Smart Init Matriochka**, optionnel sinon. Ne modifie pas r_eff agrégé. Pas de pruning des têtes endormies.
- **displacement/** — calcul du rang de déplacement pour caractériser la *classe* d'opérateur structuré (Hankel, Toeplitz, Cauchy, Vandermonde) — usage secondaire, sert d'input architectural à la phase 3.

## Entrées

- Matrices d'attention A exportées en phase 1, **provenant exclusivement du set `audit_svd`**.
- Seuils ε_target (sur A) et ε_aval (dégradation tâche), pré-enregistrés.

## Sorties

- Spectres SVD par (couche, tête, exemple).
- Table Stress-Rank Map + figures (monovariées, croisées, par couche).
- Loi de transfert ajustée + diagnostics statistiques.
- Dictionnaire SCH consommé par phase 3.
- Diagnostic de Spécialisation (figure + liste des K têtes spécialisées) si Smart Init prévu en phase 3.

## Critère de fin

Go/no-go phase 2 documenté. SCH corroborée selon monotonie, reproductibilité, utilité.

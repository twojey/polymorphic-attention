# Phase 3 — ASPLayer (Adaptive Structural Processor)

Spec : [DOC/03_phase_kernel_asp.md](../../DOC/03_phase_kernel_asp.md).

## Modules attendus

- **backbone/** — `W_base`, opérateur structuré paramétré **dérivé du dictionnaire SCH phase 2** (Toeplitz, Hankel, Cauchy, Vandermonde, ou combinaison). Pas d'import de Mamba2/S4/Linear Attention comme package complet — seulement les primitives mathématiques justifiées par phase 2 (cf. DOC/03 section 1.1).
- **matriochka/** — bases `U_mat`, `V_mat` ∈ (d, R_max) emboîtées + loss Matriochka output-based (échantillonnage S, poids w_r).
- **init/** — stratégies d'initialisation des bases :
  - `random.py` — Xavier ou orthogonale (défaut).
  - `smart.py` — extraction des top-K têtes spécialisées via Diagnostic de Spécialisation phase 2, SVD sur le set `init_phase3`, concaténation dans les premières colonnes. Ablation seulement, après que random ait passé. Pas de back-prop vers les vecteurs extraits.
- **soft_mask/** — fonction de masquage douce m_{t,i} = σ(β · (α_t · R_max − i + ½)), monotone décroissante par construction.
- **ste/** — Straight-Through Estimator pour la continuité du gradient sur l'arrondi.
- **rank_sampler/** — Gumbel-Softmax (variante stochastique), à activer en deuxième intention.
- **spectrometer_stub/** — interface du Spectromètre, instanciée en phase 4. Phase 3 utilise un stub à α_t = 1 ou α_t = r_target/R_max (distillation).
- **layer/** — assemblage `y = Backbone(x) + ΔAttn(x ; m_t)` → l'**ASPLayer**.
- **layernorm_r/** — LayerNorm (éventuellement conditionnée par r_t).
- **sanity/** — tests saturation (m_t = 1), effondrement (m_t = 0), monotonie en r_t, visualisation heatmap R_target.

## Entrées

- Recommandation R_max issue de phase 2 (raffinée à partir de l'estimation préliminaire de phase 1).
- Loi de transfert r_target = f(ω, Δ, ℋ) issue de phase 2 (utilisée pour la distillation initiale).
- Dictionnaire SCH (oriente le câblage du Backbone : structures de déplacement à favoriser).
- Si Smart Init : Diagnostic de Spécialisation des têtes (phase 2.6) + accès au set `init_phase3` (phase 1.2b).
- Stratégie d'init pré-enregistrée dans `OPS/configs/phase3/init.yaml`.

## Sorties

- Implémentation de référence de l'ASPLayer.
- Stratégie d'init exécutée (random par défaut, smart en ablation).
- Quatre sanity checks documentés (saturation, effondrement, monotonie, lissité), passés en random init d'abord.
- Si smart init exécuté : ablation random vs smart avec écart quantifié.
- Heatmap R_target obtenue par distillation, baseline pour phase 4.
- Choix de β (raideur Soft-Mask), λ_M, λ_C, stratégie d'échantillonnage S, type de LayerNorm.

## Critère de fin

Go/no-go phase 3 documenté.

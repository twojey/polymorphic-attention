# Phase 1.5 — Calibration de l'Indicateur de Stress

Spec : [DOC/01b_phase_calibration_signal.md](../../DOC/01b_phase_calibration_signal.md).

## Modules attendus

- **signals/** — implémentation des **trois** candidats (S_Surprise retiré V3, cf. DOC/01b section 1.1) :
  - `s_kl/` — KL local entre p(x_t|·) et baseline empirique global ; baseline calibré préalablement.
  - `s_grad/` — norme du gradient local par rapport à L_task (mode train uniquement).
  - `s_spectral/` — rang effectif r_eff sur fenêtre glissante K via SVD partielle randomisée (Halko et al.).
- **aggregation/** — Max-Pool par couche puis Concatenate sur la deep-stack. Sortie : vecteur de dimension L par signal.
- **bench/** — banc de test hybride 50% SSG + 50% bruit pur, mélangé par séquence.
- **correlation/** — calcul de Spearman avec bootstrap pour intervalles de confiance.
- **distillability/** — Phase 1.5b : MLP léger student de S_Spectral, mesure ρ et MSE.
- **report/** — génération de la matrice de corrélation 4×3, liste des signaux retenus, verdict de Distillabilité, choix d'agrégation finalisé.

## Entrées

- Oracle entraîné en phase 1 (consommé tel quel).
- Banc de test SSG + bruit (config OPS/).
- Baseline KL empirique global (calibré préalablement, pré-enregistré).
- Fenêtre K pour S_Spectral (pré-enregistrée).
- Stratégie de sous-échantillonnage tokens (pré-enregistrée).

## Sorties

- Matrice de corrélation 4 × 3 avec IC95 %.
- Liste signed des signaux retenus pour la phase 4, avec axes couverts.
- Si S_Spectral retenu : verdict de Distillabilité (ρ student/teacher, MSE) ou choix de fallback.
- Choix final d'agrégation cross-layer/cross-head.

## Critère de fin

Go/no-go phase 1.5 documenté. Au moins un signal valide (sensibilité > 0.70 sur ≥ 1 axe ; |ρ_ℋ| < 0.20). Sinon : arrêt du protocole.

# Templates Sprints E, F, G — Phase 4 warm-up / autonomous / Phase 5 validation

> Trois sprints partageant un format quasi identique. Un fichier par Sprint réel attendu.

## Sprint E (phase 4a warm-up Spectromètre)

- **Métadonnées** : sprint_id, date, sprint_d_checkpoint, n_epochs, lambda_spec
- **Résultats** :
  - val_acc avec ground-truth ω/Δ/ℋ
  - Spearman(r_pred, r_target) intra-train
  - convergence curves
- **Critères** :
  - val_acc ≥ 0.95 × Oracle ✅/❌
  - Spearman > 0.7 ✅/❌
- **Décision** : Continuer Sprint F ?

## Sprint F (phase 4b autonomous routing)

- **Métadonnées** : sprint_e_checkpoint, n_epochs (long), seed sweep
- **Résultats** :
  - val_acc sans signaux ground-truth
  - Spearman intra-test
  - latency de routing comparée
- **Critères** :
  - val_acc ≥ 0.90 × Oracle ✅/❌
  - routing latency raisonnable
- **Décision** : Continuer Sprint G ?

## Sprint G (phase 5 validation 5a-6c)

- **Métadonnées** : sprint_f_checkpoint, tests_to_run, seeds
- **Résultats par test** :
  - 5a identifiabilité : R_max ↔ r_target ✅/❌
  - 5b élasticité ✅/❌
  - 5c self-emergent half-truth ✅/❌
  - 5d anti-fraud ✅/❌
  - 5e OOD ✅/❌
  - 6c R_max = r_med/2 ✅/❌
- **Verdict final ASP** : GO / PARTIAL / NO-GO
- **Conditions de publication Partie 2** ...

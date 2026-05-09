# Phase 5 — Validation et Falsification

Spec : [DOC/05_phase_pareto.md](../../DOC/05_phase_pareto.md).

## Modules attendus

- **identifiability/** — banc de bruit blanc (gaussien + tokens uniformes + ℋ max SSG), mesure de la distribution de R_target.
- **elasticity/** — générateur de séquences sandwich `[bruit][structure][bruit]`, mesure du Lag de Réaction (montée et descente).
- **se_ht/** — calcul séparé de SE = Accuracy / Average_Rank (math, hardware-indépendant) et HT = Accuracy / Inference_Time (matériel). Agrégation par modèle/tâche/N.
- **baselines/** — wrappers sur Transformer plein, Mamba2, Linear Attention, Hyena/RWKV, **MoD** (Mixture of Depths, baseline principal pour la comparaison d'allocation dynamique).
- **suite/** — tâches d'évaluation : Structure-MNIST OOD, LRA, needle-in-a-haystack, induction heads, sous-échantillon LM.
- **measure/** — instrumentation FLOPs, mémoire peak, Inference_Time (mesurés, pas estimés).
- **ood/** — protocole de croisement d'axes : train récursion → eval binding (et inversement), train mixte → eval tâches algorithmiques tierces. Mesure de la robustesse de R_target hors axe vu. L'extrapolation monovariée (ω restreint → ω étendu) reste comme diagnostic secondaire.
- **ablation/** — ASPLayer avec Spectromètre gelé (m_t = 1, m_t = 0) + ablations du curriculum, du Soft-Mask, de la distillation initiale.
- **heatmap/** — visualisation de R_target sur séquences typées (livrable principal du rapport).
- **report/** — courbes consolidées + déclaration succès/échec contre 5a, 5b, 5c, 5d, 5e.

## Entrées

- ASP entraîné en phase 4.
- Définitions des comparateurs avec ressources d'entraînement appariées.
- Liste des tailles N à explorer, plages OOD, fixées *avant* tout run.

## Sorties

1. **Heatmap de Rang** — séquences typées + R_target.
2. **Tableau comparatif** — ASP vs baselines sur Accuracy, FLOPs/token, Inference_Time, SE, HT (séparés).
3. **Courbes Identifiabilité, Élasticité** — données brutes + figures.
4. **Frontière de Pareto consolidée**.
5. **Rapport OOD** — comportement de R_target en extrapolation.
6. **Rapport de Falsification** — verdict explicite par sous-test (5a–5e).

## Critère de fin

Verdict explicite par test. Le succès complet exige les cinq.

# ROADMAP — projet ASP

## Contexte de reprise (lire en premier)

**État courant** : spec **V3.5 complète et purifiée**. 19 fichiers, ~2090 lignes, 0 lien cassé. Aucun code. Stack non fixée.

**Question scientifique cadrée** : *« peut-on, avec les outils mathématiques actuels, synthétiser une attention linéaire ou superlinéaire sans trade-off majeur à partir d'oracles quadratiques ? »* — réponse OUI ou NON, les deux acceptables.

**Le protocole produit toujours une réponse utilisable** :
- **OUI** = phases 1.5 + 2 + 3 passent → architecture qui démontre la possibilité.
- **NON parce que pas observable** = phase 1.5 fail → signaux locaux n'exploitent pas la structure.
- **NON parce que pas de structure** = phase 2 SCH rejetée → l'attention quadratique n'a pas de structure low-rank reproductible.
- **NON parce que catalogue insuffisant** = phase 3 fail → la structure existe mais hors du catalogue (Toeplitz/Hankel/Cauchy/Vandermonde + compositions).

Les phases 4 et 5 répondent à la question secondaire (l'ASP-couche est-elle pratiquement utilisable). Pas obligatoire pour la question principale.

**Approche recommandée** : sprint-based, minimum viable d'abord (cf. fin de ce document, section "Stratégie de stratification").

**Prochaine action immédiate** : Stage 0.2 — sur le pod RunPod RTX 5090 :
1. Cloner le repo, exécuter `bash OPS/scripts/setup_env.sh` (test de Stage 0.5 sur la machine cible).
2. Valider les primitives mathématiques listées dans `OPS/env/STACK.md` § Primitives à valider, documenter les résultats dans `OPS/env/PRIMITIVES.md`.

Une fois Stage 0.2 validé : ouverture du Sprint 1 (phase 1 sur Structure-MNIST seul, multi-Oracle reporté à Sprint 4).

Décisions cadres prises (Sprint 1) :
- **0.1 Stack** : PyTorch 2.6+ + Lightning Fabric + Hydra + uv. Détails et justifications dans `OPS/env/STACK.md`.
- **0.3 Hardware** : VPS (édition / lecture résultats) + RunPod RTX 5090 32 GB éphémère (training). Détails dans `OPS/env/HARDWARE.md`.
- **0.4 Logging** : Weights & Biases comme outil principal, manifests YAML locaux pour traçabilité offline. Détails dans `OPS/env/LOGGING.md`.

0.2 (famille Backbone) a été retirée — le Backbone est dérivé en phase 3, pas pré-sélectionné. Le 0.2 actuel concerne la disponibilité des primitives mathématiques.

**Documents canoniques** :
- `DOC/00_vision.md` — thèse, scope, principe Discovery > Reproduction, Oracle borne sup
- `DOC/01_phase_metrologie.md` — Oracle quality gates, multi-Oracle, limites SSG axes
- `DOC/01b_phase_calibration_signal.md` — 3 signaux candidats (S_KL, S_Grad, S_Spectral)
- `DOC/02_phase_audit_spectral.md` — SCH comme distribution, batterie de tests structurels (vivante)
- `DOC/03_phase_kernel_asp.md` — Backbone dérivé, Matriochka output-based, Loss Consistency
- `DOC/04_phase_routage_budget.md` — Spectromètre, Curriculum, distillation 4a/4b avec percentile p75 + loss asymétrique
- `DOC/05_phase_pareto.md` — 5 tests phase 5 + R_max/2 (6c) + différentiel (5a.ii) + comparateurs domain-aware
- `DOC/falsifiabilite.md` — 13 règles d'application, hiérarchie des risques
- `DOC/glossaire.md` — terminologie V3.5 canonique


## Légende

- `[ ]` tâche à faire
- `[x]` tâche terminée
- ⚠ **décision utilisateur requise** (j'attends une réponse avant d'exécuter)
- 🔁 dépend du résultat d'une étape précédente
- ⏸ parallélisable avec ce qui précède
- 🚪 **gate** : passage conditionné au go/no-go d'une phase

---

## Stage 0 — Setup et décisions cadres

Tout ce qui doit être fixé **avant** d'écrire la première ligne de code.

### 0.1 — Stack ML ✅
- [x] Choix : **PyTorch 2.6+ + Lightning Fabric + Hydra + uv**
- [x] Justifié dans `OPS/env/STACK.md` (instrumentation hooks, support Blackwell, custom losses phase 3+)

### 0.2 — Primitives structurées disponibles dans la stack
- [ ] Vérifier disponibilité dans la stack retenue : convolution causale longue (FFT-based), scan parallèle générique, opérateur de Toeplitz, opérateur de Hankel, SVD batchée
- [ ] Documenter dans `OPS/env/PRIMITIVES.md`
- [ ] **Pas de pré-sélection de famille de Backbone** (Mamba, S4, etc.). Le Backbone sera *dérivé* du dictionnaire SCH phase 2, pas importé. Cf. DOC/00 section 4b (principe Discovery > Reproduction).

### 0.3 — Hardware cible ✅
- [x] Topologie : **VPS (édition + lecture résultats) + RunPod RTX 5090 32 GB éphémère (training)**
- [x] Documenté dans `OPS/env/HARDWARE.md` (CUDA 12.8, contraintes Blackwell sm_120, contraintes pod éphémère)

### 0.4 — Outil de logging ✅
- [x] Choix : **Weights & Biases** (sweeps natifs, artifacts versionnés, sync cloud compatible pod éphémère)
- [x] Justifié dans `OPS/env/LOGGING.md` (conventions de nommage, tags obligatoires, articulation pré-enregistrement)

### 0.5 — Infrastructure environnement ✅ (partiel)
- [x] `pyproject.toml` créé (deps Lightning Fabric, Hydra, W&B, etc.) avec extras `cpu`/`cuda`/`dev`
- [x] `uv.lock` généré (99 packages résolus, 8 entrées torch multi-plateforme/CUDA)
- [x] `OPS/scripts/setup_env.sh` idempotent (auto-détection GPU, install uv, sync, vérif torch, login W&B)
- [ ] **À tester sur le pod RunPod RTX 5090** — Stage 0.2 prérequis

### 0.6 — Versioning ✅
- [x] `git init` à la racine du projet (branche `master`, identité locale `synthetic-attention <nuriion01@gmail.com>`)
- [x] `.gitignore` : Python, ML (W&B, checkpoints, datasets), env (anchored), IDE, OS, exception `OPS/logs/compute_budget.md`
- [x] Premier commit `init: spec V3.5 ASP + décisions cadres Sprint 1` (23 fichiers, 2770 insertions)

### 0.7 — OPS/configs squelette ✅
- [x] `OPS/configs/phase1/`, `phase1b/`, `phase2/`, `phase3/`, `phase4/`, `phase5/` (avec `.gitkeep`)
- [x] Format YAML + composition Hydra documenté dans `OPS/configs/README.md`
- [x] `OPS/configs/manifest_template.yaml` (run_id, git_commit, hardware, stack, timing, wandb)
- [x] `OPS/logs/compute_budget.md` initialisé (estimations Sprint 1, plafonds par phase)

---

## 🚪 Stage 1 — RCP via SSG

Spec : [DOC/01_phase_metrologie.md](DOC/01_phase_metrologie.md)

### 1.1 — Construction du SSG (Structural Stress Generator)
- [ ] 🔁 Implémenter axe **ω** (récursion) sur Structure-MNIST
- [ ] Implémenter axe **Δ** (distance de dépendance)
- [ ] Implémenter axe **ℋ** (entropie de la tâche)
- [ ] Vérifier l'isolation : balayage monovarié de chaque axe sans interférence
- [ ] Implémenter les balayages croisés (ω × Δ, ω × ℋ, Δ × ℋ)
- [ ] Tests unitaires : structure générée est bien-formée, paramètres reproductibles

### 1.2 — Oracle(s) — multi-Oracle (cf. DOC/01 section 7b + 8)

**Domaines V1** (au moins deux pour le proof of concept) :
- [ ] Structure-MNIST (cœur, obligatoire)
- [ ] Code synthétique (Dyck-k étendu ou mini-DSL)
- [ ] Vision (optionnel V1, MNIST/CIFAR par patches)

**Pour chaque Oracle (par domaine), quality gates obligatoires :**
- [ ] Implémenter Transformer dense sur-dimensionné modérément
- [ ] **Pas de GQA, pas de sliding window, pas de routing** — attention dense pure
- [ ] **Context length ≥ max(Δ) du SSG** (vérifié avant entraînement)
- [ ] Flash Attention autorisé (équivalent mathématiquement)
- [ ] Configurer pour instrumentation (export A par tête × couche × exemple, sans agrégation)
- [ ] **BF16 mixed-precision** pour l'entraînement, **FP64** au moment de l'extraction des A pour la SVD
- [ ] Critère d'arrêt : **plateau de loss validation** par régime (ω, Δ, ℋ), pas un nombre d'époques fixe
- [ ] **Interdiction stricte** de modifier l'Oracle entre entraînement et extraction (pas de pruning/distillation/quantization)
- [ ] Pré-enregistrer hyperparamètres dans `OPS/configs/phase1/oracle_<domaine>.yaml`
- [ ] **Architecture identique** entre Oracles (même Transformer dense), seules les adaptations strictement nécessaires au domaine sont autorisées (taille embedding, patches vs tokens)
- [ ] **Pas de pretraining**, pas de transfert. Chaque Oracle part de zéro.

### 1.2b — Tri-partition des données par domaine (règle des trois sets)
- [ ] Pour chaque domaine : découper en `train_oracle` / `audit_svd` / `init_phase3` (proposition par défaut : 70/20/10)
- [ ] Pré-enregistrer la stratégie de split dans `OPS/configs/phase1/dataset_split_<domaine>.yaml`
- [ ] Vérifier l'absence de fuite par domaine (hash des séquences disjoints)

### 1.3 — Entraînement Oracle
- [ ] Entraînement balayage ω monovarié
- [ ] Entraînement balayage Δ monovarié
- [ ] Entraînement balayage ℋ monovarié
- [ ] Entraînement balayages croisés
- [ ] Persistence des poids dans `OPS/logs/phase1/oracle/`

### 1.4 — Métriques
- [ ] Extraction des matrices d'attention par (couche, tête, exemple)
- [ ] Calcul du rang de Hankel numérique
- [ ] Calcul de l'entropie spectrale H = -Σ p_k log p_k
- [ ] Agrégation par régime (ω, Δ, ℋ)

### 1.5 — Livrables
- [ ] Courbes monovariées rang_Hankel(ω), (Δ), (ℋ) + idem pour H_spectrale
- [ ] Cartes 2D croisées
- [ ] Estimation de la demande de capacité moyenne E[rang_Hankel]/N et E[H_spectrale]/log N
- [ ] Recommandation préliminaire de R_max
- [ ] Rapport phase 1 dans `DOC/reports/phase1_report.md`

### 1.6 — 🚪 Go/no-go phase 1
- [ ] Vérifier : il existe une portion non-triviale où rang_Hankel ≪ N **ou** H_spectrale ≪ log N
- [ ] Si **no-go** : arrêt du protocole, rapport négatif documenté, fin
- [ ] Si **go** : passage stage 1.5

---

## 🚪 Stage 1.5 — Identification Gate

Spec : [DOC/01b_phase_calibration_signal.md](DOC/01b_phase_calibration_signal.md)

### 1b.1 — Signaux candidats (3, pas 4)
- [ ] Implémenter `S_KL` (KL local + baseline empirique global pré-calibré)
- [ ] Implémenter `S_Grad` (norme du gradient local)
- [ ] Implémenter `S_Spectral` (r_eff sur fenêtre glissante K via SVD partielle randomisée)
- [ ] **S_Surprise retiré** : il aurait nécessité un Backbone proxy importé, en violation du principe Discovery > Reproduction (cf. DOC/01b section 1.1)

### 1b.3 — Agrégation
- [ ] Implémenter Max-Pool par couche sur les têtes
- [ ] Implémenter Concat sur la deep-stack
- [ ] Vérifier dimensions : sortie L scalaires par signal par token

### 1b.4 — Banc de test (par Oracle / par domaine si multi-Oracle)
- [ ] Construire dataset hybride 50% SSG variant + 50% bruit pur (pour le domaine concerné)
- [ ] Calculer les 3 signaux sur tout le banc
- [ ] Pré-enregistrer stratégie de sous-échantillonnage tokens

### 1b.5 — Mesure et critères
- [ ] Régression Spearman sur (S, ω), (S, Δ), (S, ℋ) avec bootstrap IC95%
- [ ] Construire matrice de corrélation 3×3 (× N_domaines si multi-Oracle)
- [ ] Vérifier critères : `max(ρ_ω, ρ_Δ) > 0.70` ET `|ρ_ℋ| < 0.20` par signal

### 1b.6 — Phase 1.5b — Distillabilité
- [ ] 🔁 Si S_Spectral retenu : entraîner MLP léger student
- [ ] Mesurer ρ Spearman et MSE student/teacher
- [ ] Critère : ρ > 0.85
- [ ] Si échec : choisir fallback (S_Spectral en direct ou simplification Backbone)

### 1b.7 — Livrables
- [ ] Matrice de corrélation finale
- [ ] Liste des signaux retenus avec axes couverts
- [ ] Verdict de Distillabilité
- [ ] Rapport phase 1.5 dans `DOC/reports/phase1b_report.md`

### 1b.8 — 🚪 Go/no-go phase 1.5 (**point de défaillance n°1**)
- [ ] Au moins un signal valide les deux critères sur ≥ 1 axe
- [ ] Si **no-go** : arrêt total — *l'allocation dynamique par token est une illusion statistique*
- [ ] Si **go** : passage stage 2

---

## 🚪 Stage 2 — Audit Spectral

Spec : [DOC/02_phase_audit_spectral.md](DOC/02_phase_audit_spectral.md)

### 2.1 — Implémentation
- [ ] 🔁 Pipeline SVD batché sur les matrices d'attention de phase 1, **extraites sur le set `audit_svd` uniquement**
- [ ] Cast FP64 des matrices A avant SVD
- [ ] Pas d'agrégation pré-SVD : par tête, par couche, par exemple
- [ ] Calcul de `r_eff(θ)` pour θ = 0.95 et θ = 0.99
- [ ] Pré-enregistrer ε_target (sur A) et ε_aval (dégradation tâche)

### 2.2 — Stress-Rank Map
- [ ] Agrégation par (ω, Δ, ℋ)
- [ ] Tables monovariées + cartes 2D croisées
- [ ] Décomposition par couche (vérifier concentration éventuelle)

### 2.3 — Validation SCH
- [ ] Vérifier monotonie : r_eff(ω) croît, idem Δ, ℋ
- [ ] Vérifier reproductibilité : variance entre seeds/couches/têtes < seuil pré-enregistré
- [ ] Vérifier utilité : portion non-triviale où r_eff ≪ N

### 2.4 — Loi de transfert (par Oracle / par domaine)
- [ ] ⏸ Régression sur la Stress-Rank Map → r_target_d = f_d(ω, Δ, ℋ) pour chaque domaine d
- [ ] Test forme multiplicative `a_d · ω^{α_d} · Δ^{β_d} · g_d(ℋ)`
- [ ] Diagnostics R², résidus par domaine
- [ ] **Comparaison cross-domain** : tester si les exposants (α_d, β_d) convergent (SCH universelle), divergent (domain-spécifique), ou sont mixtes (partiellement universelle)

### 2.5 — Sanity check croisé
- [ ] Vérifier sur sous-ensemble LRA que la loi de transfert n'est pas un artefact du SSG

### 2.6 — Diagnostic de Spécialisation des têtes (cf. DOC/02 section 5b)
- [ ] Pour chaque tête h : calculer `spec_h = var(r_eff_h)` à travers les régimes
- [ ] Distribution de spec_h par couche (figure)
- [ ] Liste ordonnée des K têtes les plus spécialisées
- [ ] Identifier les têtes systématiquement endormies
- [ ] Statut : **obligatoire si Phase 3 utilise Smart Init Matriochka**, optionnel sinon
- [ ] **Pas de pruning** des têtes endormies — diagnostic informatif uniquement

### 2.6b — Batterie de tests structurels (cf. DOC/02 section 5c)

**Liste ouverte — à enrichir au fil des expériences.**

Batterie A — Fitting et identification de classe :
- [ ] A.1 ε_C par classe (Toeplitz, Hankel, Cauchy, Vandermonde, Identity) par régime
- [ ] A.2 class_best = argmin_C ε_C par régime, distribution
- [ ] A.3 Composition additive `A ≈ M_1 + M_2`
- [ ] A.4 Composition multiplicative `A ≈ M_1 · M_2`

Batterie B — Analyse du résidu :
- [ ] B.1 Norme du résidu après best-fit, par régime
- [ ] B.2 SVD du résidu (low-rank caché ?)
- [ ] B.3 FFT du résidu (fréquences structurées ?)
- [ ] B.4 PCA cross-régimes du résidu (classe émergente ?)

Batterie C — Robustesse et cross-domain :
- [ ] C.1 Stabilité par tête / par couche
- [ ] C.2 Cohérence cross-domain (multi-Oracle)
- [ ] C.3 Invariance par permutation
- [ ] C.4 Décomposition en blocs

Batterie C+ — Robustesse de la paramétrisation du stress :
- [ ] C.5 PCA cross-régimes des r_eff (axes SSG capturent la difficulté ?)
- [ ] C.6 Test paramétrisation alternative (MDL ou complexité algorithmique)

Batterie D — Détection d'out-of-catalogue :
- [ ] D.1 Régimes orphelins (ε_min trop élevé même avec composition)
- [ ] D.2 Signature spectrale fréquentielle
- [ ] D.3 Eigendecomposition vs SVD (asymétrie non capturée ?)
- [ ] D.4 Test non-linéaire `A ≈ f(M)` avec f non-linéaire simple

Sortie batterie :
- [ ] Table : régime → classe dominante → ε résiduel
- [ ] Liste des régimes orphelins avec hypothèses sur classe manquante
- [ ] Carte de cohérence cross-domain
- [ ] Recommandations Backbone phase 3 (additif/composé, configurable/universel, multi-tête/simple, linéaire/non-linéaire)
- [ ] **Liste des tests ajoutés au-delà de la batterie initiale** avec motivation et critère

Protocole d'extension (vivant) :
- [ ] À chaque expérience : si un nouveau test pertinent émerge, l'ajouter à DOC/02 section 5c avec ID unique, motivation, critère
- [ ] Re-exécuter sur les régimes déjà analysés pour cohérence rétrospective

### 2.7 — Livrables
- [ ] Stress-Rank Map (table + figures)
- [ ] Loi de transfert ajustée
- [ ] Dictionnaire SCH (table régime → r_eff → choix structurel optimal)
- [ ] R_max recommandé chiffré
- [ ] Diagnostic de Spécialisation (si requis pour Phase 3)
- [ ] Rapport phase 2 dans `DOC/reports/phase2_report.md`

### 2.8 — 🚪 Go/no-go phase 2
- [ ] SCH corroborée selon les trois conditions
- [ ] Si **no-go** : SCH rejetée, arrêt
- [ ] Si **go** : passage stage 3

---

## 🚪 Stage 3 — ASPLayer

Spec : [DOC/03_phase_kernel_asp.md](DOC/03_phase_kernel_asp.md)

### 3.1 — Backbone *dérivé* du dictionnaire SCH (cf. DOC/03 section 1.1)
- [ ] 🔁 Lire le dictionnaire SCH phase 2 : quelles classes structurelles dominent les régimes faciles ?
- [ ] Selon les classes dominantes, instancier l'opérateur correspondant :
  - Toeplitz dominant → convolution causale longue paramétrée (FFT-based)
  - Hankel dominant → SSM générique (A, B, C appris **sans imposer HiPPO/selectivity**)
  - Cauchy dominant → opérateur d'interpolation rationnelle paramétré
  - Combinaison → somme ou composition des classes
- [ ] **Pas d'import de Mamba2/S4/Hyena** comme package complet. Seulement les primitives mathématiques justifiées par phase 2.
- [ ] Si SCH domain-spécifique : Backbone configurable par domaine (cf. DOC/03 section 1.2)

### 3.2 — Bases Matriochka et stratégie d'init (cf. DOC/03 section 5)
- [ ] Implémenter U_max, V_max ∈ ℝ^{d × R_max} comme paramètres statiques
- [ ] **Init aléatoire (défaut)** : Xavier ou orthogonale par bloc R_max
- [ ] Pré-enregistrer la stratégie dans `OPS/configs/phase3/init.yaml`
- [ ] Lancer les sanity checks (3.6) en random init **d'abord**

### 3.2b — Smart Init Matriochka (ablation, optionnel, à faire après random init validé)
- [ ] 🔁 Consommer le Diagnostic de Spécialisation de phase 2.6
- [ ] Extraire les top-K têtes spécialisées
- [ ] SVD de leurs A_h **sur le set `init_phase3`** (séparé de train_oracle et audit_svd)
- [ ] Concaténer top-k_h vecteurs singuliers dans U_max[:, 0:K_total], V_max[:, 0:K_total]
- [ ] Reste : init aléatoire
- [ ] Pas de back-prop vers les vecteurs extraits (figés à l'init)
- [ ] Reporter l'écart smart vs random sur qualité finale et vitesse de convergence

### 3.3 — Soft-Mask et continuité du gradient
- [ ] Implémenter Soft-Mask `m_{t,i} = σ(β · (α · R_max − i + ½))` avec β calibrable
- [ ] Implémenter STE comme alternative
- [ ] Implémenter Gumbel-Softmax (en réserve)
- [ ] Vérifier monotonie m_{t,1} ≥ m_{t,2} ≥ … par construction

### 3.4 — Losses
- [ ] Implémenter `L_matriochka` output-based : `Σ_r w_r · L_task(ASPLayer(x; mask=[1×r, 0×(R_max−r)]), y_target)`
- [ ] Implémenter `L_consistency` : `E_{r, δ}[‖y(x;r) − y(x;r+δ)‖²]`
- [ ] Stratégie d'échantillonnage S sur {1, …, R_max}, pondération w_r

### 3.5 — Assemblage ASPLayer
- [ ] `y = Backbone(x) + ΔAttn(x ; m_t)`
- [ ] LayerNorm post-addition (éventuellement conditionnée par r_t)
- [ ] Stub de Spectromètre à α_t = 1 (extension pleinement ouverte)

### 3.6 — Sanity checks
- [ ] **Saturation** : à m_t = 1 partout, atteint la borne phase 1
- [ ] **Effondrement** : à m_t = 0 partout, ≡ Backbone seul
- [ ] **Monotonie** : qualité décroît monotonement quand r_t baisse
- [ ] **Lissité** : |q(r+1) − 2q(r) + q(r−1)| < seuil pré-enregistré

### 3.7 — Livrables
- [ ] ASPLayer de référence + tests unitaires
- [ ] Heatmap R_target par distillation (baseline pour phase 4)
- [ ] Recommandation β, λ_M, λ_C, w_r, type LayerNorm
- [ ] Rapport phase 3 dans `DOC/reports/phase3_report.md`

### 3.8 — 🚪 Go/no-go phase 3
- [ ] Quatre sanity checks passés
- [ ] Si **no-go** : revoir Backbone/R_max ou rééquilibrer λ_M/λ_C
- [ ] Si **go** : passage stage 4

---

## 🚪 Stage 4 — Calibration du Spectromètre

Spec : [DOC/04_phase_routage_budget.md](DOC/04_phase_routage_budget.md)

### 4.1 — Spectromètre
- [ ] 🔁 Implémenter mini-MLP léger ou conv 1D
- [ ] Entrée : signaux validés en phase 1.5 (concat des Max-Pool layer)
- [ ] Sortie : α_t ∈ [0,1] ou m_t directement
- [ ] Vérifier monotonie de m_t par construction (sigmoïdes successives ou cumulative softmax)

### 4.2 — Loss et Curriculum
- [ ] Implémenter `L_sparsity` (forme pondérée Matriochka recommandée)
- [ ] Implémenter Curriculum de Stress 3 étages
- [ ] Pré-enregistrer 5–7 valeurs de λ_budget (log-spaced)

### 4.3 — Phase 4a (warm-up avec distillation)
- [ ] Pré-entraînement ASPLayer avec Spectromètre figé à α=1
- [ ] Activation Spectromètre + λ_distil > 0
- [ ] Lancement curriculum (étages 1 → 2 → 3) pour chaque λ_budget

### 4.4 — Transition 4a → 4b
- [ ] Vérifier convergence de L_distillation
- [ ] Vérifier corrélation Spearman R_target ↔ r_target théorique > 0.80
- [ ] Vérifier absence de plafond artificiel

### 4.5 — Phase 4b (apprentissage autonome)
- [ ] Retrait total : λ_distil = 0
- [ ] Continuation entraînement sur L_task + λ_budget · L_sparsity
- [ ] Mesure de la dérive aux N premiers steps (< 30%)

### 4.6 — Diagnostics
- [ ] Distribution de R_target par λ_budget
- [ ] Corrélation R_target ↔ stress local construit
- [ ] Construction du Diagramme de Phase
- [ ] Courbe Pareto qualité/complexité

### 4.7 — Livrables
- [ ] Spectromètre entraîné (mode 4b) par valeur de λ_budget
- [ ] Diagramme de Phase final
- [ ] Courbe Pareto λ_budget
- [ ] Rapport bascule 4a → 4b
- [ ] Rapport phase 4 dans `DOC/reports/phase4_report.md`

### 4.8 — 🚪 Go/no-go phase 4
- [ ] Courbe Pareto avec portion strictement dominante
- [ ] Diagramme de Phase croissant
- [ ] R_target corrélé au stress local construit
- [ ] Si **no-go** : retour phase 1.5 (signal de surprise) ou phase 3 (architecture)
- [ ] Si **go** : passage stage 5

---

## 🚪 Stage 5 — Validation et Falsification

Spec : [DOC/05_phase_pareto.md](DOC/05_phase_pareto.md)

### 5.1 — Test 5a — Identifiabilité (deux sous-tests conjoints)

5a.i — Anti-fraude :
- [ ] Banc bruit blanc gaussien + tokens uniformes + ℋ max SSG
- [ ] Mesure distribution R_target
- [ ] **Critère** : R_target reste au plancher
- [ ] Si **fail** : modèle frauduleux, arrêt total

5a.ii — Activation différentielle (cf. DOC/05 section 1b) :
- [ ] Construire 4 conditions : (a) bruit blanc, (b) null/empty, (c) répétition triviale, (d) input structuré
- [ ] Mesurer distribution R_target par condition
- [ ] Calculer diff_score = somme des KL divergences entre paires
- [ ] **Critère** : `diff_score > seuil_actif` ET R_target sur (d) > R_target sur (a, b, c)
- [ ] Si diff_score < seuil_silent : Spectromètre passif/silencieux → fail

### 5.2 — Test 5b — Élasticité
- [ ] Générateur séquences sandwich [bruit][structure][bruit]
- [ ] Mesure courbe R_target(t)
- [ ] Calcul Lag de Réaction
- [ ] **Critère** : Lag < seuil pré-enregistré ET descente symétrique

### 5.3 — Test 5c — SE et HT
- [ ] Mesure Accuracy par modèle/tâche
- [ ] Mesure Average_Rank ASP
- [ ] Mesure Inference_Time réelle (pas analytique)
- [ ] Calcul SE = Accuracy / Average_Rank
- [ ] Calcul HT = Accuracy / Inference_Time
- [ ] **Critère** : SE ≥ 2× Transformer ET HT ≥ SSM pur

### 5.4 — Comparateurs domain-aware (Test 5d — Pareto, cf. DOC/05 section 4)

**Pour le sequence-domain** (Structure-MNIST séquentialisé, code, langue) :
- [ ] Transformer plein, Mamba2, Linear Attention, Hyena/RWKV, **MoD** (baseline principal)

**Pour le vision-domain** (si évalué) :
- [ ] ViT, ConvNet (ResNet ou ConvNeXt), MLP-Mixer

**Pour le code-domain** (si évalué séparément) :
- [ ] Transformer plein, MoD

**Règles communes** :
- [ ] Entraînement à ressources strictement appariées (FLOPs, données, seeds)
- [ ] **Pas de comparaison cross-domain** : ne pas comparer un ASP-vision à un Mamba (sequence-only)
- [ ] **Reporting honnête de l'init** : si l'ASP utilise Smart Init, reporter aussi l'ASP en random init (cf. DOC/05 section 6b)

### 5.5 — Suite d'évaluation
- [ ] Structure-MNIST OOD
- [ ] LRA (Long Range Arena)
- [ ] Needle-in-a-haystack
- [ ] Induction heads
- [ ] Sous-échantillon LM (WikiText / sous-ensemble Pile)
- [ ] 3 seeds × N ∈ {2¹⁰, 2¹², 2¹⁴, 2¹⁶}

### 5.6 — Test 5d — Pareto
- [ ] Mesure FLOPs/token, mémoire peak, latence pour chaque (modèle, tâche, N)
- [ ] Construction frontière de Pareto
- [ ] **Critère** : ASP sur la frontière, dominance stricte sur ≥ 1 famille de tâches
- [ ] Test décisif vs MoD : ASP montre montée graduée de R_target sur récursion progressive ?

### 5.6b — Test 6c — R_max réduit (ASP dépasse l'Oracle ?, cf. DOC/05 section 6c)
- [ ] Calculer `r_med = médiane(r_eff_oracle | régime moyen)`
- [ ] Entraîner ASPLayer avec `R_max = r_med / 2`
- [ ] Évaluer qualité sur la suite phase 5
- [ ] **Critère succès strict** : qualité ≥ 95% Oracle → ASP a structurellement dépassé l'Oracle
- [ ] **Critère succès partiel** : qualité ∈ [80%, 95%] → ASP compétitif, pas strictement supérieur
- [ ] Statut : optionnel pour Pareto strict, **obligatoire pour toute claim "ASP plus efficace que l'Oracle"**

### 5.7 — Test 5e — OOD croisé
- [ ] Train récursion (ω varié, Δ minimal) → eval binding (Δ varié, ω minimal)
- [ ] Train binding → eval récursion (inversion)
- [ ] Train mixte → eval tâches algorithmiques tierces
- [ ] **Critère** : R_target s'élève sur axe inédit, pas figé sur axe vu

### 5.8 — Ablations
- [ ] ASPLayer avec Spectromètre gelé à m=1
- [ ] ASPLayer avec Spectromètre gelé à m=0
- [ ] Sans Curriculum
- [ ] Sans distillation 4a
- [ ] Sans Loss Consistency

### 5.9 — Livrables
- [ ] Heatmap de Rang (visualisation R_target sur séquences typées)
- [ ] Tableau comparatif ASP vs baselines (Accuracy, FLOPs, Inference_Time, SE, HT)
- [ ] Courbes 5a, 5b avec données brutes
- [ ] Frontière de Pareto consolidée
- [ ] Rapport OOD
- [ ] Rapport ablations
- [ ] Rapport final dans `DOC/reports/phase5_report.md`

### 5.10 — 🚪 Verdict final
- [ ] Vérifier conjonction stricte des 5 sous-tests (5a, 5b, 5c, 5d, 5e)
- [ ] Si succès complet : preuve du concept ASP, écriture article
- [ ] Si succès partiel : rapport documenté, retour amont ou clôture

---

## Tracks transverses (continus)

### T.1 — Pré-enregistrement
- [ ] Avant chaque phase, fixer tous les seuils dans `OPS/configs/phaseX/thresholds.yaml`
- [ ] Aucun ajustement post-hoc autorisé

### T.2 — Suivi compute
- [ ] Tracker GPU-hours par phase dans `OPS/logs/compute_budget.md`
- [ ] Alerter si > 2× le budget initial estimé

### T.3 — Documentation maintenance
- [ ] À chaque décision technique : mettre à jour la doc concernée
- [ ] À chaque écart par rapport à la spec : justifier dans le rapport de phase

### T.4 — Mémoire (Claude)
- [ ] Mettre à jour `project_asp_overview.md` à chaque jalon majeur
- [ ] Sauver les décisions techniques durables (stack, hardware, etc.)

---

## Repères de sortie anticipée

Trois points où le projet peut s'arrêter proprement avec un résultat publiable :

1. **Après phase 1.5 no-go** — papier court : *« L'allocation dynamique par token n'est pas observable sur ce banc »*. Résultat négatif honnête.
2. **Après phase 2 no-go** — papier moyen : *« La SCH est rejetée — la matrice d'attention n'est pas low-rank de manière reproductible sur Structure-MNIST »*.
3. **Après phase 4 succès partiel** — papier long sur l'ASPLayer comme architecture, sans la prétention d'efficience polymorphe. Pareto pas démontré, mais kernel existe.

Ces sorties sont des résultats, pas des échecs. Documentées dès maintenant pour ne pas céder à la tentation de continuer à perte.

---

## Stratégie de stratification — sprint-based, minimum viable

Le protocole complet est ambitieux (compute lourd, multi-Oracle, multi-tâche, multi-N, ablations). Pour éviter le piège "tout ou rien", l'exécuter en sprints où chaque sprint a un livrable autonome :

### Sprint 1 — Fondations (≈ 3–4 semaines)
- Stage 0 complet (stack, hardware, logging, env, git, configs squelette).
- Stage 1 minimum : **un seul Oracle** sur Structure-MNIST (multi-Oracle reporté à Sprint 4 ou plus tard).
- Stage 1.5 : 3 signaux sur ce seul Oracle.
- **Décision après Sprint 1** : si phase 1.5 passe, Sprint 2. Sinon → rapport négatif court, fin du projet ou pivot.

### Sprint 2 — Audit minimal (≈ 4–6 semaines)
- Stage 2 minimum : **batteries A + B uniquement** (fitting + résidu). Batteries C, D, C+ reportées.
- Stage 3 : ASPLayer en random init, sanity checks 1–4.
- **Décision après Sprint 2** : SCH corroborée (au moins faiblement) ET architecture saine ? Oui → Sprint 3. Non → rapport phase 2 ou 3 publiable, fin ou pivot.

### Sprint 3 — Polymorphisme minimal (≈ 6–8 semaines)
- Stage 4 sur Structure-MNIST seul, curriculum simplifié, λ_budget = 1 valeur.
- Test 5a.i + 5a.ii (Identifiabilité + différentiel).
- **Décision après Sprint 3** : Diagramme de Phase croissant + Spectromètre actif ? Oui → Sprint 4. Non → rapport phase 4 publiable.

### Sprint 4 — Validation et expansion (≈ 8–12 semaines)
- Multi-Oracle (ajout d'au moins un domaine secondaire : code synthétique).
- Phase 5 complète : 5b, 5c, 5d (avec MoD), 5e croisé, 6c R_max/2.
- Ablations.
- **Décision après Sprint 4** : verdict final.

À chaque sprint, **un rapport autonome publiable**. Ce qui transforme un projet 12–18 mois "tout ou rien" en série de mini-projets, chacun produisant de l'information.

### Cette stratification ne change pas le protocole

Tous les seuils, critères go/no-go, conditions sont les mêmes. C'est juste l'ordre et l'étendue d'exécution. La rigueur falsifiable est préservée — on n'allège pas les tests, on les exécute à plus petit périmètre puis on étend.

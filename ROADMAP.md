# ROADMAP — projet ASP

## Contexte de reprise (lire en premier)

**État courant (mai 2026)** : spec V3.5 complète + **pré-prep VPS maximale terminée**. 7+ commits sur `master`. **119 tests CPU passent** (~7s). Stack arrêtée. Aucun run sur GPU encore — tout attend l'accès au pod RunPod RTX 5090.

**Question scientifique cadrée** : *« peut-on, avec les outils mathématiques actuels, synthétiser une attention linéaire ou superlinéaire sans trade-off majeur à partir d'oracles quadratiques ? »* — réponse OUI ou NON, les deux acceptables.

**Le protocole produit toujours une réponse utilisable** :
- **OUI** = phases 1.5 + 2 + 3 passent → architecture qui démontre la possibilité.
- **NON parce que pas observable** = phase 1.5 fail → signaux locaux n'exploitent pas la structure.
- **NON parce que pas de structure** = phase 2 SCH rejetée → l'attention quadratique n'a pas de structure low-rank reproductible.
- **NON parce que catalogue insuffisant** = phase 3 fail → la structure existe mais hors du catalogue (Toeplitz/Hankel/Cauchy/Vandermonde + compositions).

Les phases 4 et 5 répondent à la question secondaire (l'ASP-couche est-elle pratiquement utilisable). Pas obligatoire pour la question principale.

**Approche recommandée** : sprint-based, minimum viable d'abord (cf. fin de ce document, section "Stratégie de stratification").

**Prochaine action immédiate** : 4 bloquants externes (remote git, MLflow tmux/systemd, clé SSH pod→VPS), puis sur le pod RunPod RTX 5090 :
1. Cloner le repo, exécuter `bash OPS/scripts/setup_env.sh` (test final de Stage 0.5 sur Blackwell).
2. Stage 0.2 : `PYTHONPATH=CODE uv run python OPS/scripts/validate_primitives.py`. Coller la sortie dans `OPS/env/PRIMITIVES.md`.
3. Sprint 1 phase 1 : `PYTHONPATH=CODE uv run python -m phase1_metrologie.run --config-path=../../OPS/configs/phase1 --config-name=oracle_smnist`.

**Pré-prep VPS maximale terminée** : 119 tests CPU passent (~7s).
- Phase 1 (SSG + Oracle + extraction + métriques + sweeps + report) ✅
- Phase 1.5 (3 signaux + agrégation + bench + Spearman + distillabilité + driver) ✅
- Phase 2 (SVD pipeline + Stress-Rank Map + loi de transfert + spec heads + batteries A/B/D + driver) ✅
- Phase 3 STRUCTUREL (Soft-Mask + STE + Matriochka + L_matriochka + L_consistency + ASPLayer + sanity checks + Backbone abstrait) ✅
- Phase 4 STRUCTUREL (Spectromètre + Curriculum + distillation V3.5 p75 asymétrique + L_sparsity + Diagramme Phase + Pareto λ) ✅
- Phase 5 STRUCTUREL (5a anti-fraude + 5a.ii différentiel + 5b lag + 5c SE/HT + 5d Pareto + 5e OOD + 6c R_max/2) ✅
- Configs Hydra phase 1/1.5/2/3/4/5 pré-enregistrées ✅
- Templates rapports phase 1/1.5/2/3/4/5 ✅
- Tout ce qui dépend de résultats spéculatifs (Backbone concret, choix signaux retenus, comparateurs phase 5.4) reste laissé en `TODO post-phase-X` — Discovery > Reproduction respecté.

Décisions cadres prises (Sprint 1) :
- **0.1 Stack** : PyTorch ≥ 2.11.0+cu128 (seul build sm_120 Blackwell) + Lightning Fabric + Hydra + uv. Détails et justifications dans `OPS/env/STACK.md`. Aligné avec stack Lumis validée sur 5090 (`/root/lumis/OPS`).
- **0.3 Hardware** : VPS (édition / serveur MLflow / lecture résultats) + RunPod RTX 5090 32 GB éphémère (training). Image Docker `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`, Cloud type COMMUNITY, volume 100 GB. Détails dans `OPS/env/HARDWARE.md`.
- **0.4 Logging** : MLflow self-hosted sur le VPS (zéro compte externe, aucune donnée ne sort des machines), accès pod via tunnel SSH `ssh -L 5000:127.0.0.1:5000`. Manifests YAML locaux pour traçabilité offline. Détails dans `OPS/env/LOGGING.md`.

0.2 (famille Backbone) a été retirée — le Backbone est dérivé en phase 3, pas pré-sélectionné. Le 0.2 actuel concerne la disponibilité des primitives mathématiques (`OPS/scripts/validate_primitives.py` prêt, 6 checks CPU passent, à ré-exécuter sur 5090).

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
- `DOC/reports/{phase1,phase1b,phase2,phase3,phase4,phase5}_template.md` — squelettes rapports

**Drivers Hydra prêts** :
- `phase1_metrologie.run` — Oracle SMNIST + extraction + métriques (Sprint 1)
- `phase1b_calibration_signal.run` — 3 signaux + Spearman + verdict gate (Sprint 1)
- `phase1b_calibration_signal.run_distillability` — sub-phase 1.5b (Sprint 1)
- `phase2_audit_spectral.run` — SVD + SRM + transfer law + batteries (Sprint 2)

**Drivers à écrire post-pod** (dépendent de résultats spéculatifs) :
- `phase3_kernel_asp.run` — après que phase 2 ait dicté la classe SCH dominante
- `phase4_routage_budget.run` — après que phase 1.5 ait dicté les signaux retenus
- `phase5_pareto.run` — après que phase 3 ait produit l'ASPLayer concrète (les fonctions de test sont déjà testées et prêtes à être branchées)


## 🚀 Démarrage session suivante (à l'arrivée du pod)

**Lis CETTE section en premier au reboot.** Elle te dit exactement quoi faire pour reprendre sans frottement.

### Étape 1 — Bloquants externes (sans pod, ~15 min)

Faisables sur le VPS avant même d'avoir le pod :

1. **Push le repo sur un remote git privé**
   ```bash
   cd /root/polymorphic-attention
   git remote add origin <url>          # GitHub privé / Gitea / etc.
   git push -u origin master
   ```
2. **Lance MLflow server sur le VPS dans tmux/systemd**
   ```bash
   tmux new -s mlflow
   bash OPS/scripts/start_mlflow_server.sh
   # Ctrl-B D pour détacher
   ```
   Vérification : `curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5000/` → `200`
3. **Configure une clé SSH du pod vers le VPS** (pour le tunnel MLflow). Génère sur le pod, copie la pubkey dans `~/.ssh/authorized_keys` du VPS.

### Étape 2 — Sur le pod (≈ 30 min de setup)

1. **Crée le pod RunPod**
   - GPU : `NVIDIA GeForce RTX 5090`
   - Cloud type : `COMMUNITY`
   - Image : `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`
   - Volume : 100 GB, Container disk : 40 GB
   - Port : 22/tcp

2. **Clone + setup**
   ```bash
   git clone <url> /workspace/polymorphic-attention
   cd /workspace/polymorphic-attention
   bash OPS/scripts/setup_env.sh                # idempotent, install uv + deps
   ```
   Si l'index `cu128` n'existe pas → bascule `cu126` ou `cu124` dans `pyproject.toml § [[tool.uv.index]]`, regénère `uv.lock` (`uv lock`), commit, repush, repull.

3. **Tunnel SSH vers le VPS pour MLflow**
   ```bash
   ssh -N -f -L 5000:127.0.0.1:5000 user@<vps_ip>
   export MLFLOW_TRACKING_URI=http://localhost:5000
   ```

4. **Stage 0.2 — validation primitives Blackwell**
   ```bash
   PYTHONPATH=CODE uv run python OPS/scripts/validate_primitives.py
   ```
   Cherche : `is_blackwell: True`, `device_capability: (12, 0)`, 6/6 checks ✅. Colle la sortie dans `OPS/env/PRIMITIVES.md`. Commit.

### Étape 3 — Sprint 1 (≈ 30 GPU-h prévues)

```bash
PYTHONPATH=CODE uv run python -m phase1_metrologie.run \
    --config-path=../../OPS/configs/phase1 --config-name=oracle_smnist
```

Le driver enchaîne : SSG tri-partition → Oracle dense Transformer → extraction A FP64 sur audit_svd → Hankel + entropie spectrale par régime → MLflow + manifest. Sortie attendue : un run MLflow `phase1_s1_smnist_oracle_<short_hash>` consultable depuis le VPS.

Puis phase 1.5 :
```bash
PYTHONPATH=CODE uv run python -m phase1b_calibration_signal.run \
    --config-path=../../OPS/configs/phase1b --config-name=signals \
    oracle_checkpoint=<path/to/oracle.ckpt>
```

### Étape 4 — Décision Sprint 1 → Sprint 2

Si phase 1.5 GO → phase 2 (driver `phase2_audit_spectral.run` déjà prêt).
Si phase 1.5 NO-GO → rapport négatif court (cf. "Repères de sortie anticipée"), fin du protocole.

### Pièges connus à anticiper

- Index PyTorch `cu128` peut nécessiter bascule `cu126`/`cu124` (cf. STACK.md § Évolutions)
- Driver NVIDIA < 555 → warning Blackwell. Vérifier `nvidia-smi` au démarrage du pod
- MLflow OSS sans auth → bind 127.0.0.1 obligatoire, accès via tunnel SSH
- Aucun run "registered" possible si `git status` est dirty (cf. `shared.runner.git_status_clean`)

### Documents à consulter en cas de doute

- `OPS/env/STACK.md` — stack ML, primitives à valider, contraintes Blackwell
- `OPS/env/HARDWARE.md` — topologie pod, ENV vars Blackwell, pièges 5090 (issus Lumis)
- `OPS/env/LOGGING.md` — MLflow self-hosted, tunnel SSH, conventions runs
- `OPS/configs/README.md` — pré-enregistrement, conventions configs
- `DOC/falsifiabilite.md` — règles d'application du protocole
- `DOC/glossaire.md` — terminologie V3.5 canonique

---

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
- [x] Choix : **PyTorch ≥ 2.11.0+cu128 + Lightning Fabric + Hydra + uv** (seul build sm_120 — confirmé via stack Lumis `/root/lumis/OPS`)
- [x] Justifié dans `OPS/env/STACK.md` (instrumentation hooks, support Blackwell, custom losses phase 3+)
- [x] ENV vars Blackwell auto-set par `OPS/scripts/setup_env.sh` (TORCH_CUDA_ARCH_LIST, PYTORCH_CUDA_ALLOC_CONF, MAX_JOBS, etc.)

### 0.2 — Primitives structurées disponibles dans la stack ✅ partiel
- [x] Script `OPS/scripts/validate_primitives.py` : 6 checks (svd FP64 batchée, svd_lowrank, fft.rfft, lstsq batché, vmap, attention dense + extraction A FP64)
- [x] Validation CPU sur VPS : 6/6 passent
- [ ] **À exécuter sur RTX 5090 pour Stage 0.2 final** — coller le résultat dans `OPS/env/PRIMITIVES.md`
- [x] **Pas de pré-sélection de famille de Backbone** (Mamba, S4, etc.). Le Backbone sera *dérivé* du dictionnaire SCH phase 2, pas importé. Cf. DOC/00 section 4b (principe Discovery > Reproduction).

### 0.3 — Hardware cible ✅
- [x] Topologie : **VPS (édition + lecture résultats) + RunPod RTX 5090 32 GB éphémère (training)**
- [x] Documenté dans `OPS/env/HARDWARE.md` (CUDA 12.8, contraintes Blackwell sm_120, contraintes pod éphémère)

### 0.4 — Outil de logging ✅
- [x] Choix : **MLflow self-hosted sur le VPS** (zéro compte externe, accès pod via tunnel SSH, sweeps via Hydra multirun)
- [x] Justifié dans `OPS/env/LOGGING.md` (topologie, conventions de nommage, tags obligatoires, articulation pré-enregistrement)
- [x] `OPS/scripts/start_mlflow_server.sh` (lance MLflow sur 127.0.0.1:5000 avec backend SQLite + filesystem artifacts)

### 0.5 — Infrastructure environnement ✅ (partiel)
- [x] `pyproject.toml` créé (deps Lightning Fabric, Hydra, MLflow, etc.) avec extras `cpu`/`cuda`/`dev`
- [x] `uv.lock` généré (résolution réussie, entrées torch multi-plateforme/CUDA)
- [x] `OPS/scripts/setup_env.sh` idempotent (auto-détection GPU, install uv, sync, vérif torch, vérif MLFLOW_TRACKING_URI)
- [ ] **À tester sur le pod RunPod RTX 5090** — Stage 0.2 prérequis

### 0.6 — Versioning ✅
- [x] `git init` à la racine du projet (branche `master`, identité locale `synthetic-attention <nuriion01@gmail.com>`)
- [x] `.gitignore` : Python, ML (mlruns, checkpoints, datasets), env (anchored), IDE, OS, exception `OPS/logs/compute_budget.md`
- [x] Premier commit `init: spec V3.5 ASP + décisions cadres Sprint 1` (23 fichiers, 2770 insertions)

### 0.7 — OPS/configs squelette ✅
- [x] `OPS/configs/phase1/`, `phase1b/`, `phase2/`, `phase3/`, `phase4/`, `phase5/` (avec `.gitkeep`)
- [x] Format YAML + composition Hydra documenté dans `OPS/configs/README.md`
- [x] `OPS/configs/manifest_template.yaml` (run_id, git_commit, hardware, stack, timing, mlflow)
- [x] `OPS/logs/compute_budget.md` initialisé (estimations Sprint 1, plafonds par phase)

---

## 🚪 Stage 1 — RCP via SSG

Spec : [DOC/01_phase_metrologie.md](DOC/01_phase_metrologie.md)

### 1.1 — Construction du SSG (Structural Stress Generator) ✅ V1 token-level
- [x] Axe **ω** (récursion) — `CODE/phase1_metrologie/ssg/structure_mnist.py`
- [x] Axe **Δ** (distance de dépendance)
- [x] Axe **ℋ** (entropie de la tâche)
- [x] Balayages monovariés via `sweep_monovariate(axis, values, base)`
- [ ] Balayages croisés (ω × Δ, ω × ℋ, Δ × ℋ) — TODO Sprint 1 sur le pod
- [x] Tests unitaires : 11 tests SSG (reproductibilité, isolation des axes, tri-partition)
- ⚠ Limite V1 : token-level (digits=tokens 0-9, pas pixels MNIST). Bascule pixel-level prévue post-Sprint 1 si besoin.

### 1.2 — Oracle(s) — multi-Oracle (cf. DOC/01 section 7b + 8)

**Domaines V1** (au moins deux pour le proof of concept) :
- [x] Structure-MNIST (cœur, obligatoire) — code prêt
- [ ] Code synthétique (Dyck-k étendu ou mini-DSL) — **reporté Sprint 4**
- [ ] Vision (optionnel V1, MNIST/CIFAR par patches) — **reporté Sprint 4 ou plus tard**

**Pour chaque Oracle (par domaine), quality gates obligatoires :** ✅ code prêt pour SMNIST
- [x] Transformer dense sur-dimensionné modérément — `CODE/phase1_metrologie/oracle/transformer.py`
- [x] **Pas de GQA, pas de sliding window, pas de routing** — `DenseAttention` pure
- [x] **Context length ≥ max(Δ) du SSG** — `max_seq_len: 4096` config (à vérifier vs sweep effectif)
- [x] Flash Attention via `scaled_dot_product_attention` à l'entraînement, désactivé à l'extraction
- [x] Configurer pour instrumentation (export A par tête × couche × exemple, sans agrégation) — `AttentionExtractor`
- [x] **BF16 mixed-precision** entraînement (`Fabric(precision="bf16-mixed")`), **FP64** extraction
- [x] Critère d'arrêt : **plateau de loss validation** — implémenté dans `train_oracle()` (patience + tolerance)
- [ ] Vérification plateau **par régime** (ω, Δ, ℋ), pas seulement en moyenne — TODO sur le pod
- [x] **Interdiction stricte** de modifier l'Oracle entre entraînement et extraction — pas de pruning/distillation/quantization dans le code
- [x] Hyperparamètres pré-enregistrés dans `OPS/configs/phase1/oracle_smnist.yaml`
- [x] Tests Oracle (forward, mask PAD, capture FP64, smoke training) — 11 tests CPU passent
- [ ] **Architecture identique** entre Oracles (à vérifier au Sprint 4 quand multi-Oracle)
- [x] **Pas de pretraining**, from scratch — config impose

### 1.2b — Tri-partition des données par domaine (règle des trois sets) ✅ code prêt
- [x] Découpage `train_oracle` / `audit_svd` / `init_phase3` (70/20/10) — `phase1_metrologie.ssg.split_indices`
- [x] Pré-enregistré dans `OPS/configs/phase1/dataset_split_smnist.yaml`
- [x] Index disjoints garantis par construction (testé, 4 tests passent)
- [ ] Hash des séquences pour audit a posteriori — TODO post-extraction sur le pod (logger dans manifest)

### 1.3 — Entraînement Oracle (à exécuter sur le pod)
- [x] Driver `phase1_metrologie.run` enchaîne sweep + train + extract + métriques
- [x] Sweep monovarié ω implémenté (`sweep_monovariate(axis="omega", ...)`)
- [x] Sweep monovarié Δ implémenté
- [x] Sweep monovarié ℋ implémenté
- [x] Sweep croisé implémenté (`crossed_sweep(axis1, axis2, ...)`) — pas branché dans run.py V1, à brancher au pod
- [ ] Exécution sur le pod (~30 GPU-h prévues, plafond 60)
- [ ] Persistence des poids via MLflow artifacts

### 1.4 — Métriques ✅ code prêt
- [x] Extraction des matrices d'attention par (couche, tête, exemple) FP64 — `CODE/phase1_metrologie/oracle/extract.py`
- [x] Rang de Hankel numérique — `CODE/phase1_metrologie/metrics/hankel.py`
- [x] Entropie spectrale H = -Σ p_k log p_k — `CODE/phase1_metrologie/metrics/spectral.py`
- [x] Tests sur matrices synthétiques (rank-1 outer product → 0, uniforme → log N, exponential signal → Hankel rank 1, etc.)
- [ ] Agrégation par régime (ω, Δ, ℋ) — TODO post-extraction sur le pod

### 1.5 — Livrables
- [x] Helpers plot prêts (`shared.plotting.monovariate_curve`, `heatmap_2d`, `stress_rank_map`)
- [x] Aggregation par régime prête (`shared.aggregation.aggregate_by_regime` + `aggregate_by_regime_2d`)
- [x] Recommandation R_max heuristique prête (`phase1_metrologie.report.recommend_r_max` : 1.5 × max p90)
- [x] Génération rapport Markdown prête (`phase1_metrologie.report.render_markdown_report`)
- [x] Template `DOC/reports/phase1_template.md`
- [ ] Génération effective post-run sur le pod
- [ ] Estimation E[rang_Hankel]/N et E[H_spectrale]/log N — calcul direct depuis stats agrégées

### 1.6 — 🚪 Go/no-go phase 1
- [ ] Vérifier : il existe une portion non-triviale où rang_Hankel ≪ N **ou** H_spectrale ≪ log N
- [ ] Si **no-go** : arrêt du protocole, rapport négatif documenté, fin
- [ ] Si **go** : passage stage 1.5

---

## 🚪 Stage 1.5 — Identification Gate

Spec : [DOC/01b_phase_calibration_signal.md](DOC/01b_phase_calibration_signal.md)

### 1b.1 — Signaux candidats (3, pas 4) ✅ code prêt
- [x] `S_KL` (KL local + baseline empirique global pré-calibré) — `CODE/phase1b_calibration_signal/signals/s_kl.py`
- [x] `S_Grad` (norme du gradient local) — `CODE/phase1b_calibration_signal/signals/s_grad.py`
- [x] `S_Spectral` (r_eff sur fenêtre glissante K via SVD partielle randomisée) — `CODE/phase1b_calibration_signal/signals/s_spectral.py`
- [x] **S_Surprise retiré** : violation Discovery > Reproduction (cf. DOC/01b §1.1)

### 1b.3 — Agrégation ✅
- [x] Max-Pool par couche sur les têtes + Concat deep-stack — `signals/aggregation.py:aggregate_signal_per_token`
- [x] Tests : `aggregate_signal_per_token((L, B, H, N))` → `(B, L, N)`

### 1b.4 — Banc de test (par Oracle / par domaine si multi-Oracle) ✅ code prêt
- [x] Dataset hybride 50% SSG variant + 50% bruit pur — `bench/hybrid.py`
- [x] Calcul des 3 signaux : pipeline en place dans le driver run.py
- [x] Sous-échantillonnage tokens (tous les 8) pré-enregistré dans `signals.yaml`

### 1b.5 — Mesure et critères ✅ code prêt
- [x] Régression Spearman + bootstrap IC95 — `bench/spearman.py:bootstrap_spearman_ci`
- [x] Matrice de corrélation 3×3 — `bench/spearman.py:signal_correlations`
- [x] Critères 0.70 / 0.20 — `bench/spearman.py:passes_phase1b_criteria`
- [x] Tests : faux signaux corrélés → passent, bruit pur → ne passent pas

### 1b.6 — Phase 1.5b — Distillabilité ✅ code prêt
- [x] StudentMLP + entraînement — `bench/distillability.py:train_student`
- [x] Mesure ρ Spearman + MSE relative
- [x] Critère ρ > 0.85 + MSE_relative < 0.5 (pré-enregistrés dans thresholds_phase1b.yaml)
- [ ] Fallback à choisir si échec (cf. DOC/01b §5)

### 1b.7 — Livrables
- [x] Helpers : `correlation_matrix_heatmap` dans `shared.plotting`
- [x] Driver `phase1b_calibration_signal.run` produit la matrice + verdict + tags MLflow `retained_signals`
- [x] Template `DOC/reports/phase1b_template.md`
- [ ] Génération effective post-run sur le pod
- [ ] Verdict de Distillabilité — driver `run_distillability` prêt

### 1b.8 — 🚪 Go/no-go phase 1.5 (**point de défaillance n°1**)
- [ ] Au moins un signal valide les deux critères sur ≥ 1 axe
- [ ] Si **no-go** : arrêt total — *l'allocation dynamique par token est une illusion statistique*
- [ ] Si **go** : passage stage 2

---

## 🚪 Stage 2 — Audit Spectral

Spec : [DOC/02_phase_audit_spectral.md](DOC/02_phase_audit_spectral.md)

### 2.1 — Implémentation ✅ code prêt
- [x] Pipeline SVD batché — `phase2_audit_spectral.svd_pipeline.svd_attention`
- [x] Cast FP64 des matrices A avant SVD — built-in
- [x] Pas d'agrégation pré-SVD : par (couche, tête, exemple)
- [x] Calcul de `r_eff(θ)` pour θ = 0.95 et θ = 0.99 — `r_eff_from_singular_values`
- [x] ε_target / ε_aval pré-enregistrés dans `OPS/configs/phase2/thresholds_phase2.yaml`
- [ ] Exécution sur les matrices A de phase 1 (post-pod)

### 2.2 — Stress-Rank Map ✅ code prêt
- [x] Agrégation par (ω, Δ, ℋ) — `build_monovariate_srm`
- [x] Tables monovariées + cartes 2D croisées — `build_2d_srm`, `median_grid`, `iqr_grid` + `shared.plotting.stress_rank_map`
- [x] V3.5 distributionnelle : médiane + IQR + p10/p90/p99 (cf. `shared.aggregation.RegimeStats`)
- [ ] Décomposition par couche (post-extraction)

### 2.3 — Validation SCH
- [x] Monotonie automatique via `transfer_law` (signe des exposants α, β)
- [x] Reproductibilité via comparaison IQR/médiane (seuils dans `thresholds_phase2.yaml`)
- [x] Utilité via portion régimes avec r_eff ≪ N — calculable depuis SRM
- [ ] Vérifications effectives post-run

### 2.4 — Loi de transfert ✅ code prêt
- [x] Régression log-linéaire `log r = log a + α log(1+ω) + β log(1+Δ) + γ·ℋ` — `phase2_audit_spectral.transfer_law.fit_transfer_law`
- [x] Diagnostics R², résidus, n — dans `TransferLawFit`
- [x] **Comparaison cross-domain** : `cross_domain_compare` retourne verdict `universal | partial | domain_specific` (CV des exposants)
- [x] Tests : retrouve les exposants vrais à 5% sur données synthétiques
- [ ] Application multi-Oracle (Sprint 4)

### 2.5 — Sanity check croisé
- [ ] Vérifier sur sous-ensemble LRA — TODO Sprint 4

### 2.6 — Diagnostic de Spécialisation des têtes ✅ code prêt
- [x] Calcul `spec_h = var(r_eff_h)` — `phase2_audit_spectral.head_specialization.diagnose_heads`
- [x] Liste ordonnée top-K — `top_specialized_heads`
- [x] Détection têtes endormies — `is_dormant` flag
- [x] **Pas de pruning** — code informatif uniquement, conforme spec
- [ ] Distribution figure (à brancher avec `shared.plotting`)
- [ ] Statut : **obligatoire si Phase 3 utilise Smart Init Matriochka** — déclenchera execution

### 2.6b — Batterie de tests structurels (cf. DOC/02 section 5c)

**Liste ouverte — à enrichir au fil des expériences.**

Batterie A — Fitting et identification de classe : ✅ code prêt
- [x] A.1 ε_C par classe (Toeplitz, Hankel, Identity) par régime — `phase2_audit_spectral.batteries.battery_a.fit_class`
- [x] A.2 class_best = argmin_C ε_C par régime — `fit_classes_per_regime` retourne `BatteryAResult.class_best`
- [x] A.3 Composition additive `A ≈ M_1 + M_2` — `fit_additive_composition`
- [ ] A.4 Composition multiplicative `A ≈ M_1 · M_2` — V2 (non trivial)
- [ ] Cauchy / Vandermonde — V2 (paramétrisation low-rank requise)

Batterie B — Analyse du résidu : ✅ code prêt
- [x] B.1 Norme du résidu après best-fit — `BatteryBResult.norm_residual`
- [x] B.2 SVD du résidu (low-rank caché ?) — `analyze_residual_svd`
- [x] B.3 FFT du résidu (fréquences structurées ?) — `analyze_residual_fft`
- [x] B.4 PCA cross-régimes du résidu — `pca_cross_regime_residuals`

Batterie C — Robustesse et cross-domain : **reportée Sprint 4 (multi-Oracle)**
- [ ] C.1 Stabilité par tête / par couche
- [ ] C.2 Cohérence cross-domain (multi-Oracle)
- [ ] C.3 Invariance par permutation
- [ ] C.4 Décomposition en blocs

Batterie C+ — Robustesse de la paramétrisation du stress : **reportée**
- [ ] C.5 PCA cross-régimes des r_eff (axes SSG capturent la difficulté ?)
- [ ] C.6 Test paramétrisation alternative (MDL ou complexité algorithmique)

Batterie D — Détection d'out-of-catalogue : ✅ code prêt
- [x] D.1 Régimes orphelins — `detect_orphan_regimes` (seuil ε_min > 0.30 même avec composition)
- [x] D.3 Eigendecomposition vs SVD (asymétrie) — `eigen_svd_asymmetry`
- [x] Driver `battery_d_analysis` génère hypothèses sur classes manquantes
- [ ] D.2 Signature spectrale fréquentielle — V2 (extension via FFT du résidu, déjà partiellement dans batterie B)
- [ ] D.4 Test non-linéaire `A ≈ f(M)` — V2

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
- [x] Driver `phase2_audit_spectral.run` produit Stress-Rank Map + transfer law fit + diagnostic spec heads + batteries A/B/D
- [x] Template `DOC/reports/phase2_template.md`
- [ ] Dictionnaire SCH (table régime → r_eff → classe dominante) — calculable depuis output driver
- [ ] R_max recommandé chiffré (raffiné depuis phase 1)
- [ ] Génération effective post-run sur le pod

### 2.8 — 🚪 Go/no-go phase 2
- [ ] SCH corroborée selon les trois conditions
- [ ] Si **no-go** : SCH rejetée, arrêt
- [ ] Si **go** : passage stage 3

---

## 🚪 Stage 3 — ASPLayer

Spec : [DOC/03_phase_kernel_asp.md](DOC/03_phase_kernel_asp.md)

### 3.1 — Backbone *dérivé* du dictionnaire SCH (cf. DOC/03 section 1.1)
- [x] Interface abstraite `Backbone` (ABC) — `CODE/phase3_kernel_asp/backbone.py`
- [x] `IdentityBackbone` + `LinearBackbone` pour tests structurels
- [ ] 🔁 **Backbone concret** dérivé post-phase-2 selon classe SCH dominante :
  - Toeplitz dominant → convolution causale longue paramétrée (FFT-based) — **à écrire**
  - Hankel dominant → SSM générique (A, B, C appris sans HiPPO) — **à écrire**
  - Cauchy dominant → opérateur d'interpolation rationnelle — **à écrire**
  - Combinaison → composite — **à écrire**
- [x] **Pas d'import de Mamba2/S4/Hyena** — pas dans deps, pas dans code
- [ ] Si SCH domain-spécifique : Backbone configurable par domaine

### 3.2 — Bases Matriochka et stratégie d'init ✅ code prêt
- [x] U_max, V_max ∈ ℝ^{d × R_max} comme paramètres statiques — `MatriochkaBases`
- [x] Init Xavier (défaut) + orthogonale par bloc — `MatriochkaInitConfig`
- [x] Pré-enregistré dans `OPS/configs/phase3/asp_layer.yaml`
- [x] Tests : init normaux + smart init préserve les colonnes

### 3.2b — Smart Init Matriochka ✅ code prêt
- [x] Consomme `smart_init_vectors: torch.Tensor` (calculé depuis spec heads phase 2.6)
- [x] Concaténation top-K dans U_max[:, 0:K] + V_max[:, 0:K]
- [x] Reste : init aléatoire
- [x] Freeze des K colonnes via `register_hook` sur le gradient (`freeze_smart_columns()`)
- [ ] Reporting smart vs random — TODO post-phase-2

### 3.3 — Soft-Mask et continuité du gradient ✅ code prêt
- [x] Soft-Mask `m_{t,i} = σ(β · (α · R_max − i + ½))` — `phase3_kernel_asp.soft_mask.soft_mask`
- [x] STE alternative — `hard_threshold_ste`
- [x] Gumbel-Softmax en réserve — `gumbel_softmax_mask`
- [x] Monotonie m_{t,1} ≥ m_{t,2} ≥ … testée

### 3.4 — Losses ✅ code prêt
- [x] `L_matriochka` output-based — `phase3_kernel_asp.losses.loss_matriochka`
- [x] `L_consistency` — `loss_consistency`
- [x] Stratégie d'échantillonnage S — `matriochka_rank_schedule` (R_max + R_max/2 + R_max/4 + random)
- [x] Pondération w_r — `matriochka_weights` (uniform | decreasing)

### 3.5 — Assemblage ASPLayer ✅ code prêt
- [x] `y = Backbone(x) + ΔAttn(x ; m_t)` + LayerNorm — `phase3_kernel_asp.asp_layer.ASPLayer`
- [x] Trois interfaces de forward : `forward_with_alpha`, `forward_with_rank`, `forward_with_mask`
- [x] Stub Spectromètre à α_t = 1 → `forward_with_alpha(alpha=torch.ones(...))`

### 3.6 — Sanity checks ✅ code prêt
- [x] **Saturation** — `sanity.sanity_check_monotone_quality` + comparaison oracle
- [x] **Effondrement** (r=0 ≡ Backbone) — `sanity.sanity_check_collapse`
- [x] **Monotonie** q(r) — `sanity_check_monotone_quality`
- [x] **Lissité** — `sanity_check_smoothness`
- [x] Driver harnais : `run_all_sanity_checks` retourne `SanityResult`

### 3.7 — Livrables
- [x] ASPLayer + tests unitaires (16 tests CPU passent)
- [x] Configs `OPS/configs/phase3/asp_layer.yaml` (tous seuils pré-enregistrés)
- [x] Template `DOC/reports/phase3_template.md`
- [ ] Driver `phase3_kernel_asp.run` — **à écrire** post-phase-2 (intègre Backbone concret)
- [ ] Heatmap R_target par distillation (baseline phase 4) — `forward_with_rank(R_max)` sur grille de stress + `shared.plotting.heatmap_2d`

### 3.8 — 🚪 Go/no-go phase 3
- [ ] Quatre sanity checks passés
- [ ] Si **no-go** : revoir Backbone/R_max ou rééquilibrer λ_M/λ_C
- [ ] Si **go** : passage stage 4

---

## 🚪 Stage 4 — Calibration du Spectromètre

Spec : [DOC/04_phase_routage_budget.md](DOC/04_phase_routage_budget.md)

### 4.1 — Spectromètre ✅ code prêt
- [x] `Spectrometer` (MLP léger configurable, `input_dim` à fixer post-phase-1.5) — `phase4_routage_budget.spectrometer.Spectrometer`
- [x] `FrozenAlphaSpectrometer` (α=1 partout, pour 4a warm-up) — `FrozenAlphaSpectrometer`
- [x] Sortie α_t ∈ [0,1] via sigmoid (mode "alpha") ou logits (mode "logits")
- [ ] Instanciation concrète post-phase-1.5 avec `input_dim = L × n_signaux retenus`
- [ ] Monotonie cumulative softmax (V2, optionnelle)

### 4.2 — Loss et Curriculum ✅ code prêt
- [x] `L_sparsity` pondérée Matriochka — `sparsity_loss.loss_sparsity` (uniform/linear/exponential)
- [x] Curriculum de Stress 3 étages — `curriculum.CurriculumScheduler` + `default_curriculum()`
- [x] 7 valeurs de λ_budget (log-spaced) pré-enregistrées dans `OPS/configs/phase4/spectrometer.yaml`

### 4.3 — Phase 4a (warm-up avec distillation V3.5) ✅ code prêt
- [x] Cible distillation : **percentile p75** du r_eff Oracle — `compute_p75_targets`
- [x] Loss asymétrique (γ ∈ [0.1, 0.3]) — `asymmetric_distillation_loss`
- [x] FrozenAlphaSpectrometer pour pré-entraînement à α=1
- [x] Curriculum stages avec transition par val_acc + min_steps
- [ ] Driver `phase4_routage_budget.run` — **à écrire** (consomme signaux retenus phase 1.5)

### 4.4 — Transition 4a → 4b ✅ code prêt
- [x] `TransitionMonitor` vérifie : convergence loss + ρ Spearman R_pred ↔ R_target théorique > 0.80 + variance R_pred > seuil (anti-plafond)
- [x] Critères pré-enregistrés dans `OPS/configs/phase4/spectrometer.yaml`
- [x] Tests : passe quand convergé ET corrélé, échoue si pas assez d'historique

### 4.5 — Phase 4b (apprentissage autonome)
- [x] λ_distil = 0 dans config (retrait total)
- [x] L_task + λ_budget · L_sparsity → `loss_sparsity` prêt
- [ ] Mesure de la dérive aux N premiers steps — TODO dans driver

### 4.6 — Diagnostics ✅ code prêt
- [x] Distribution de R_target par λ_budget — Hydra multirun + agrégation MLflow
- [x] Construction du Diagramme de Phase — `diagram_phase.build_phase_diagram` + `is_phase_diagram_increasing`
- [x] Courbe Pareto qualité/complexité — `build_pareto_curve` (frontière non dominée)

### 4.7 — Livrables
- [x] Configs `OPS/configs/phase4/spectrometer.yaml` (curriculum, λ_budget, γ asymétrique, transition)
- [x] Tests : 16 tests CPU passent
- [x] Template `DOC/reports/phase4_template.md`
- [ ] Driver `phase4_routage_budget.run` — **à écrire** post-phase-3 (instancie Spectrometer + ASPLayer concrète)

### 4.8 — 🚪 Go/no-go phase 4
- [ ] Courbe Pareto avec portion strictement dominante
- [ ] Diagramme de Phase croissant
- [ ] R_target corrélé au stress local construit
- [ ] Si **no-go** : retour phase 1.5 (signal de surprise) ou phase 3 (architecture)
- [ ] Si **go** : passage stage 5

---

## 🚪 Stage 5 — Validation et Falsification

Spec : [DOC/05_phase_pareto.md](DOC/05_phase_pareto.md)

### 5.1 — Test 5a — Identifiabilité (deux sous-tests conjoints) ✅ code prêt

5a.i — Anti-fraude :
- [x] Banc bruit blanc — `phase5_pareto.test_5a_identifiability.run_anti_fraud`
- [x] Mesure distribution R_target via interface `ASPLayerEvaluable`
- [x] Critère "R_target reste au plancher" — `floor_threshold` configurable
- [x] Tests : passes sur layer factice non-pathologique, fail sur layer "frauduleux" qui fire sur du bruit

5a.ii — Activation différentielle ✅ code prêt
- [x] 4 conditions (noise / null / trivial / structured) — argument `conditions` dict
- [x] Distribution R_target par condition + KL pairwise — `run_differential_activation`
- [x] diff_score sum KL + critère structured > others
- [x] Configs pré-enregistrées dans `OPS/configs/phase5/stress_test.yaml`

### 5.2 — Test 5b — Élasticité ✅ code prêt
- [x] Détection lag rise/fall — `phase5_pareto.test_5b_elasticity.detect_lag`
- [x] Run sur sandwich [bruit][structure][bruit] — `run_elasticity_test`
- [x] Critère lag < seuil + symétrie

### 5.3 — Test 5c — SE et HT ✅ code prêt
- [x] Mesure Inference_Time réelle (warmup + N appels + CUDA sync) — `measure_inference_time`
- [x] SE = Acc/Avg_Rank, HT = Acc/Time — `compute_se_ht`
- [x] Critère SE ≥ 2× Transformer ET HT ≥ SSM — `passes_se_ht_targets`

### 5.4 — Comparateurs domain-aware (Test 5d — Pareto, cf. DOC/05 section 4)

- [x] Interface Pareto + frontière — `phase5_pareto.pareto.{ModelEvaluation, pareto_frontier, asp_on_frontier}`
- [ ] **Comparateurs concrets** sequence-domain à instancier post-phase-3 :
  - [ ] Transformer plein (peut réutiliser Oracle phase 1)
  - [ ] MoD (baseline principal — DOC/05 §4)
  - [ ] Mamba2, Linear Attention, Hyena/RWKV — selon classe SCH retenue (si non-Hankel par exemple, Mamba devient pertinent comme baseline et non comme Backbone importé)
- [ ] Vision-domain : ViT, ConvNet, MLP-Mixer (si évalué Sprint 5)
- [ ] Code-domain : Transformer plein, MoD (si évalué Sprint 4-5)

**Règles communes** (codifiées dans configs) :
- [x] Pas de cross-domain comparison (`pareto_frontier` filtre par `domain`)
- [ ] Entraînement à ressources strictement appariées — TODO driver phase 5
- [ ] Reporting smart vs random init — TODO driver phase 5

### 5.5 — Suite d'évaluation
- [x] Configuration pré-enregistrée : 3 seeds × N ∈ {2¹⁰, 2¹², 2¹⁴, 2¹⁶} dans `stress_test.yaml`
- [x] Liste benchmarks : Structure-MNIST OOD, LRA, Needle-in-a-haystack, Induction heads, LM subsample
- [ ] **Datasets concrets à intégrer** post-pod : LRA download, induction heads generator, LM subsample

### 5.6 — Test 5d — Pareto ✅ code prêt
- [x] Frontière de Pareto — `pareto_frontier` (filtre les points dominés)
- [x] ASP sur la frontière par domaine — `asp_on_frontier`
- [x] Tests : dominé exclu, frontière croît en qualité avec rang
- [ ] Mesure FLOPs/token, mémoire peak, latence — instrumentation post-phase-3

### 5.6b — Test 6c — R_max réduit ✅ code prêt
- [x] `compute_r_med_oracle` — médiane des r_eff Oracle
- [x] `evaluate_rmax_half` retourne verdict `strict (≥ 95%) | partial (∈ [80, 95]) | fail`
- [x] Tests sur les trois cas
- [x] Configs : `enabled: true` par défaut dans `stress_test.yaml`

### 5.7 — Test 5e — OOD croisé ✅ code prêt
- [x] `run_ood_test` mesure ratio R_target_eval / R_target_train
- [x] Critère elevation_threshold ≥ 1.2 pré-enregistré
- [ ] Datasets train/eval croisés à générer post-phase-1 (sweep récursion vs binding)

### 5.8 — Ablations
- [x] Liste pré-enregistrée dans `stress_test.yaml`:
  - asp_with_alpha_frozen_one (utilise `FrozenAlphaSpectrometer`)
  - asp_with_alpha_frozen_zero (Spectro retournant zéros)
  - asp_no_curriculum (skip CurriculumScheduler)
  - asp_no_distillation_4a (skip phase 4a, démarre direct 4b)
  - asp_no_loss_consistency (skip `loss_consistency` dans 3.4)
- [ ] Drivers d'ablation à brancher post-phase-3/4

### 5.9 — Livrables
- [x] Template `DOC/reports/phase5_template.md`
- [x] Plot helpers (Heatmap de Rang via `shared.plotting.stress_rank_map`)
- [ ] Driver `phase5_pareto.run` — **à écrire** post-phase-3+4 (instancie ASPLayer concrète + comparateurs)

### 5.10 — 🚪 Verdict final
- [ ] Vérifier conjonction stricte des 5 sous-tests (5a, 5b, 5c, 5d, 5e)
- [ ] Si succès complet : preuve du concept ASP, écriture article
- [ ] Si succès partiel : rapport documenté, retour amont ou clôture

---

## Tracks transverses (continus)

### T.1 — Pré-enregistrement ✅ partiel
- [x] Tous les seuils phase 1, 1.5, 2, 3, 4, 5 commités dans `OPS/configs/phaseX/{thresholds_phaseX,...}.yaml` AVANT exécution
- [x] `shared.runner.git_status_clean()` impose un arbre git propre pour qu'un run soit `status:registered`. Sinon → `exploratory` (jamais cité comme preuve).
- [x] `shared.mlflow_helpers.set_status_invalidated()` permet de retag un run a posteriori si la config a été modifiée
- [ ] À chaque phase exécutée : passer en revue thresholds.yaml AVANT le run, signer git, lancer

### T.2 — Suivi compute
- [x] `OPS/logs/compute_budget.md` initialisé (estimations Sprint 1, plafonds par phase)
- [ ] Édition manuelle après chaque run de référence
- [ ] Alerte si > 2× budget — à surveiller manuellement (pas d'automatisation)

### T.3 — Documentation maintenance
- [x] Templates rapports `DOC/reports/phase{1,1b,2,3,4,5}_template.md`
- [ ] À chaque run : remplir le rapport correspondant, commit
- [ ] À chaque écart spec : justifier dans le rapport

### T.4 — Mémoire (Claude)
- [x] `project_asp_overview.md` à jour (sortie de pré-prep)
- [x] `project_stack_decisions.md` à jour (MLflow + alignement Lumis)
- [ ] Mise à jour à chaque jalon de phase

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

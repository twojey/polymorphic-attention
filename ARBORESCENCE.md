# Arborescence du projet — Attention Superlinéaire Polymorphe (ASP)

**Convention de base** : arborescence racine figée en trois dossiers majuscules `CODE/`, `DOC/`, `OPS/`.

---

## 📋 Vue d'ensemble

```
polymorphic-attention/
├── CODE/                    # Code source Python (catalog + sprints + livrables + 6 phases + infra)
├── DOC/                     # Documentation Markdown (FR)
├── OPS/                     # Configs Hydra + scripts shell + logs + checkpoints
├── pyproject.toml           # uv + deps Python
├── uv.lock                  # Lockfile déterministe
├── ROADMAP.md               # État courant + prochaines actions
├── ARBORESCENCE.md          # Ce fichier
└── [.git, .venv, .claude]   # Infra (non-doc)
```

---

## 📁 CODE/ — Code source

### Vue d'ensemble

```
CODE/
├── shared/                  # Primitives réutilisables (cross-module)
├── infra/                   # Machine profile / hardware detection
├── catalog/                 # ⭐ Catalogue Partie 1 — 131 Properties / 23 familles (complet)
├── sprints/                 # ⭐ Orchestrateurs Sprint (B/C/D/E/F/G/S4-S7)
├── livrables/               # ⭐ Génération artefacts scientifiques (paper, tables)
├── phase1_metrologie/       # Phase 1 — Oracle SMNIST + extraction
├── phase1b_calibration_signal/  # Phase 1.5 — calibration signaux S_KL/Grad/Spectral
├── phase2_audit_spectral/   # Phase 2 — SVD + SCH dictionnaire
├── phase3_kernel_asp/       # Phase 3 — ASPLayer entraînement
├── phase4_routage_budget/   # Phase 4 — Spectromètre + curriculum
└── phase5_pareto/           # Phase 5 — tests validation 5a-6c
```

### `CODE/shared/` — Modules partagés

```
shared/
├── checkpoint.py            # Checkpoint atomique (save/load par key, fingerprint)
├── retry.py                 # ⭐ @retry décorateur + retry_call (backoff exp + jitter)
├── logging_helpers.py       # setup_logging UTC + log_exceptions + log_checkpoint
├── mlflow_helpers.py        # MLflow log opt-in (graceful si URI absent)
├── plotting.py              # Helpers matplotlib (heatmaps, courbes)
├── aggregation.py           # Agrégation cross-batteries (V3.5 distributional)
├── runner.py                # Runner générique phases
└── tests/                   # tests unitaires
```

**Responsabilité** : zéro code métier, infrastructure réutilisable cross-Oracle/cross-phase.

### `CODE/infra/` — Hardware abstraction

```
infra/
├── machine.py               # MachineProfile : auto-détect GPU/CPU, dtype_svd
└── tests/
```

### `CODE/catalog/` — Catalogue Partie 1 (LIVRABLE PRIORITAIRE)

```
catalog/
├── run.py                   # CLI : python -m catalog.run --oracle X --level Y
├── report.py                # Génération rapport Markdown
├── cross_oracle.py          # Comparaison signatures cross-Oracle
│
├── properties/              # 131 Properties classifiées 23 familles (A-W + N) — catalogue complet
│   ├── base.py              # Property abstract + PropertyContext + scope Literal
│   ├── registry.py          # PropertyRegistry singleton + @register_property
│   ├── family_a_spectral/   # 7 : A1-A7 (r_eff, stable_rank, condition, entropy, decay, PR, d_eff Rényi-2)
│   ├── family_b_structural/ # 10 : B0-B9 (identity, Toeplitz, Hankel, Cauchy, sparse, banded, ...)
│   ├── family_c_token_stats/ # 9 : C1-C9 (KL, entropies, Fisher, Wasserstein, S_Grad)
│   ├── family_d_geometric/  # 6 : D1-D6 (cosine, baseline, subspaces, frobenius rnd, nuclear, Grassmann)
│   ├── family_e_information/ # 5 : E1-E5 (mutual, compressibility, redundancy, MDL, entropy rate)
│   ├── family_f_dynamic/    # 4 : F1-F4 (Lipschitz, temporal, Jacobian softmax, Lyapunov proxy)
│   ├── family_g_algebraic/  # 8 : G1-G8 (trace/det, symmetry, idempotence, charpoly, B-S, D-module, syzygies)
│   ├── family_h_cross_layer/ # 6 : H1-H6 (residual, composition, r_eff traj, convergence, deep residual, sink)
│   ├── family_i_cross_head/ # 5 : I1-I5 (diversity, specialization, clustering, agreement, redundancy)
│   ├── family_j_markov/     # 5 : J1-J5 (convergence, stationary, mixing, reversibility, spectral gap)
│   ├── family_k_graph/      # 5 : K1-K5 (Laplacian, persistent homology, PageRank, modularity, Cheeger)
│   ├── family_l_frequency/  # 5 : L1-L5 (FFT2D, wavelets, periodicity, DCT-II, spectral peaks)
│   ├── family_m_conditional/ # 4 : M1-M4 (token type, stress, input dep, token class sensitivity)
│   ├── family_n_comparative/ # 4 : N1-N4 (F-div, preservation, Lipschitz diff, prediction agreement)
│   ├── family_o_displacement/ # 5 : O1-O5 (Toeplitz/Cauchy/Vandermonde rank, Kailath, Pan)
│   ├── family_p_realization/ # 7 : P1-P7 (Hankel real., HSV, decay, min order, hier blocks, gramians, Markov real.)
│   ├── family_q_hierarchical/ # 6 : Q1-Q6 (H-matrix, HSS, nestedness×2, H-dist, off-diagonal rank)
│   ├── family_r_rkhs/       # 6 : R1-R6 (Mercer, RFF, Bochner, truncated, Gauss test, Nyström spectrum)
│   ├── family_s_tensors/    # 5 : S1-S5 (Tucker, Tensor Train, Hier Tucker, CP-rank ALS, pair unfoldings)
│   ├── family_t_equivariance/ # 4 : T1-T4 (permutation, subgroup, cyclic, reflection)
│   ├── family_u_sparse_structured/ # 5 : U1-U5 (Butterfly, Monarch, Block-sparse, Pixelfly, Sparse+LR)
│   ├── family_v_operators/  # 5 : V1-V5 (pseudo-diff, CZ decay, compactness, commutator, Schatten p)
│   └── family_w_logic/      # 5 : W1-W5 (pattern, dep proxy, NIP/VC-shatter, NIP score, VC proxy)
│
├── oracles/                 # Adapters Oracle (interface AbstractOracle)
│   ├── base.py              # AbstractOracle + AttentionDump + RegimeSpec
│   ├── _minimal_transformer.py  # ⭐ MinimalTransformer GPT-style (fallback Oracles)
│   ├── synthetic.py         # SyntheticOracle (génération random softmax)
│   ├── smnist.py            # SMNISTOracle (phase 1 wrapper)
│   ├── language.py          # ⭐ LLOracle + HFLanguageBackend + MinimalLMBackend + nested_parentheses
│   ├── vision.py            # ⭐ VisionOracle + HFVisionBackend + MinimalViTBackend + PatchTokenizer
│   └── code.py              # ⭐ CodeOracle + MinimalCodeBackend + Dyck-k parser/validator
│
├── batteries/               # Composition Property × Oracle → résultats
│   ├── base.py              # Battery + BatteryResults (avec n_workers parallel dispatch)
│   └── levels.py            # level_minimal/principal/extended/full/research
│
├── projectors/              # Structures matricielles paramétrées
│   ├── base.py              # AbstractProjector
│   ├── toeplitz.py          # ProjectorToeplitz
│   ├── hankel.py            # ProjectorHankel
│   ├── cauchy.py            # ProjectorCauchy
│   ├── banded.py            # ProjectorBanded
│   ├── block_diagonal.py    # ProjectorBlockDiagonal
│   ├── butterfly.py         # ProjectorButterfly
│   ├── monarch.py           # ProjectorMonarch
│   └── pixelfly.py          # ProjectorPixelfly
│
├── fast_solvers/            # ⭐ Algorithmes rapides pour matrices structurées
│   ├── levinson.py          # Levinson-Durbin Toeplitz solver (scipy backed)
│   ├── cauchy.py            # Cauchy matrix multiply + solve (référence)
│   ├── displacement.py      # Sylvester displacement + generators (Kailath)
│   └── tests/
│
└── tests/                   # Tests intégration end-to-end catalog
```

### `CODE/sprints/` — ⭐ Orchestrateurs Sprint

```
sprints/
├── base.py                  # SprintBase (abstract) + SprintResult + SprintStatus + manifest
├── run.py                   # CLI dispatcher : python -m sprints.run --sprint B
│
├── sprint_b_re_extract.py   # Sprint B : re-extraction dumps phase 1 V2 (retry intégré)
├── sprint_c_catalog_full.py # Sprint C : Battery research × dumps Sprint B
├── sprint_d_phase3_v3.py    # Sprint D : phase 3 V3+ avec Backbone informé Sprint C
├── sprint_e_phase4_warmup.py     # Sprint E : phase 4a warm-up Spectromètre
├── sprint_f_phase4_autonomous.py # Sprint F : phase 4b autonomous routing
├── sprint_g_phase5_validation.py # Sprint G : phase 5 tests 5a-6c (verdict ASP)
├── sprint_s4_smnist_extended.py  # Sprint S4 : SMNIST seq_len 1024-4096
├── sprint_s5_vision.py      # Sprint S5 : Vision Oracle (DINOv2 ou MinimalViT)
├── sprint_s6_code.py        # Sprint S6 : Code Oracle (Dyck-k + StarCoder)
├── sprint_s7_ll.py          # Sprint S7 : LL Oracle (Llama-3.2-1B + TinyStories)
└── tests/                   # Sprint base + robustesse end-to-end
```

Garanties Sprint :
- Checkpoint/resume atomique (FileNotFoundError si fingerprint mismatch)
- Logging horodaté UTC vers `<output_dir>/sprint.log`
- Manifest reproductible (git hash + dirty flag, torch, python, cuda)
- MLflow logger opt-in
- Critères go/no-go explicites (skip_if_failed=True déclenche SKIPPED)
- Retry sur extract_regime (via shared/retry.py)

### `CODE/livrables/` — ⭐ Génération artefacts scientifiques

```
livrables/
├── cross_oracle_synthesis.py     # Table cross-Oracle Property × Oracle + variance
├── partie1_signatures.py         # Signatures textuelles per Oracle (15 heuristiques)
├── partie1_predictions_vs_measured.py  # Confrontation paris a priori
├── partie2_asp_verdict.py        # Verdict ASP GO/PARTIAL/NO-GO (5a + 5c + 6c mandatory)
├── paper_figures.py              # Heatmap signatures + barplot predictions + Pareto
├── run_all.py                    # ⭐ Orchestrateur 1-shot : tous livrables Partie 1
└── tests/                        # 16 tests livrables
```

### `CODE/phase1_metrologie/` — Phase 1 (Oracle SMNIST)

```
phase1_metrologie/
├── run.py / run_extract.py      # Drivers Hydra
├── report.py                    # Rapport phase 1
├── sweeps.py                    # Configs sweep (ω, Δ, ℋ)
├── metrics/, oracle/, ssg/      # Métriques + Oracle + Signature Stochastic Generator
└── tests/
```

### `CODE/phase1b_calibration_signal/` — Phase 1.5

```
phase1b_calibration_signal/
├── run.py / run_distillability.py
├── signals/                     # S_KL, S_Grad, S_Spectral
├── bench/                       # Benchmarks robustesse
└── tests/
```

### `CODE/phase2_audit_spectral/` — Phase 2

```
phase2_audit_spectral/
├── run.py
├── svd_pipeline.py              # SVD batched FP64
├── head_specialization.py
├── signal_decoupling.py
├── stress_rank_map.py           # SRM V3.5
├── transfer_law.py              # Loi r ~ ω^α · Δ^β
├── batteries/
└── tests/
```

### `CODE/phase3_kernel_asp/` — Phase 3

```
phase3_kernel_asp/
├── run_train.py
├── asp_layer.py                 # ASPLayer drop-in attention
├── backbone.py / backbone_concrete.py
├── transformer.py
├── losses.py                    # Loss ASP-spécifiques
├── matriochka.py                # Nested representations
├── smart_init.py                # Init informée Sprint C
├── soft_mask.py                 # Masque progressif
├── sanity.py
└── tests/
```

### `CODE/phase4_routage_budget/` — Phase 4

```
phase4_routage_budget/
├── run.py
├── spectrometer.py              # Mini-réseau inférence r_target
├── curriculum.py                # Curriculum warm-up → autonomous
├── sparsity_loss.py / distillation.py / diagram_phase.py
└── tests/
```

### `CODE/phase5_pareto/` — Phase 5

```
phase5_pareto/
├── run.py
├── abstract.py / pareto.py
├── test_5a_identifiability.py   # 5a
├── test_5b_elasticity.py        # 5b
├── test_5c_se_ht.py             # 5c
├── test_5e_ood.py               # 5e
├── test_6c_rmax_half.py         # 6c
└── tests/
```

---

## 📚 DOC/ — Documentation (Markdown, français)

```
DOC/
├── INTRODUCTION.md              # Point d'entrée unique
├── 00_FONDATIONS.md             # Thèse ASP + vocabulaire architectural/math
├── CATALOGUE.md                 # ⭐ 131 propriétés × 23 catégories + Oracle selection + prédictions
├── falsifiabilite.md            # Critères go/no-go par phase
├── carnet_de_bord.md            # Journal chronologique
│
├── 01_phase_metrologie.md       # Phase 1 spec
├── 01b_phase_calibration_signal.md  # Phase 1.5 spec
├── 02_phase_audit_spectral.md   # Phase 2 spec
├── 03_phase_kernel_asp.md       # Phase 3 spec
├── 04_phase_routage_budget.md   # Phase 4 spec
├── 05_phase_pareto.md           # Phase 5 spec
│
├── adr/                         # Architecture Decision Records
│
├── sprints/                     # ⭐ Sprints docs
│   └── README.md                # Index 10 Sprints + lancement
│
├── reports/                     # Rapports phase + Sprint
│   ├── README.md
│   ├── phase{1,1b,2,3,4,5}_template.md
│   ├── phase2.md                # Instancié
│   └── sprints/                 # Templates rapports Sprint B-G + S4-S7
│       ├── sprint_b_template.md
│       ├── sprint_c_template.md
│       ├── sprint_d_template.md
│       ├── sprint_efg_template.md
│       └── sprint_s4s7_template.md
│
└── paper/                       # ⭐ Outlines paper Partie 1 + 2
    ├── README.md                # Guide auto-génération via livrables
    ├── bibliography.bib         # BibTeX (Kailath, Hackbusch, Mercer, etc.)
    ├── partie1/
    │   ├── outline.md           # Plan paper "Mathematical Signatures"
    │   ├── predictions_a_priori.yaml  # 15 paris pré-enregistrés
    │   └── figures/
    └── partie2/
        ├── outline.md           # Plan paper "ASP Verdict"
        └── figures/
```

---

## ⚙️ OPS/ — Configs + scripts + logs + checkpoints

```
OPS/
├── configs/                     # Configs Hydra par phase + sprint + catalog
│   ├── catalog/                 # base.yaml, smnist_*, cross_oracle.yaml
│   ├── sprints/                 # ⭐ base.yaml + sprint_{b,c,d,e,f,g,s4,s5,s6,s7}.yaml
│   ├── phase1/ … phase5/        # Configs Hydra par phase
│   └── manifest_template.yaml   # Manifest run reproductibilité
│
├── env/                         # Env documentation (runtime)
│   ├── HARDWARE.md              # VPS + pod, ENV vars Blackwell sm_120
│   ├── STACK.md                 # PyTorch ≥ 2.11+cu128 + Fabric + Hydra + uv
│   ├── LOGGING.md               # Conventions MLflow self-hosted
│   └── PRIMITIVES.md            # Checklist primitives mathématiques GPU
│
├── setup/                       # Scripts de bootstrap (= "from zero to running")
│   ├── README.md
│   ├── SETUP.md                 # Pod RTX 5090 step-by-step
│   ├── POD_CPU_SETUP.md         # Pod CPU pour phase 1.5
│   ├── launch_extract.sh        # Phase 1 extract
│   ├── launch_phase1b.sh        # Phase 1.5
│   ├── launch_phase2.sh         # Phase 2
│   ├── launch_phase3.sh         # Phase 3
│   └── launch_sprint.sh         # ⭐ Sprint B-G + S4-S7 générique (nohup + watch)
│
├── scripts/                     # Scripts d'installation pure
│   ├── setup_pod.sh / setup_pod_cpu.sh / setup_env.sh
│   ├── install_uv.sh / install_python_deps.sh
│   ├── blackwell_env.sh         # Variables Blackwell sm_120
│   ├── verify_torch.sh / validate_primitives.py
│   ├── start_mlflow_server.sh / start_mlflow_tunnel.sh
│   ├── monitor_run.sh
│   └── lib/common.sh            # Strict mode, traps ERR, asp::init_logging
│
├── checkpoints/                 # Checkpoints modèles + Sprint
│   ├── oracle_e2f0b5e.ckpt      # Oracle SMNIST entraîné
│   └── sprints/                 # Checkpoints atomique par Sprint
│
├── logs/                        # Logs persistent horodatés
│   ├── mlflow/                  # MLflow artifacts
│   ├── sprints/                 # ⭐ sprint_{ID}_{UTC}.log + output dirs
│   └── manifests/               # Manifests JSON d'exécution
│
└── secrets/                     # API keys (.gitignored)
```

---

## 📊 Fichiers racine

```
polymorphic-attention/
├── pyproject.toml               # uv + deps
├── uv.lock
├── ROADMAP.md                   # État courant + prochaines actions
├── ARBORESCENCE.md              # Ce fichier
├── .gitignore
└── .claude/                     # Config Claude Code (local)
```

---

## 🔄 Convention arborescence — Immuabilité racine

Les trois dossiers `CODE/`, `DOC/`, `OPS/` sont **figés** au niveau racine :

- ✅ Ajouter modules *à l'intérieur* de `CODE/`
- ✅ Ajouter documentation *à l'intérieur* de `DOC/`
- ✅ Ajouter configs/scripts *à l'intérieur* de `OPS/`
- ❌ Créer nouveaux dossiers au niveau racine
- ❌ Déplacer CODE/DOC/OPS de la racine

---

## 🎯 État du projet (2026-05-12 fin session)

| Bloc | État | Notes |
|---|---|---|
| **Catalog Properties** | ✅ **131 / 131 (100 %)** | 23 familles A-W + N, catalogue complet |
| **Catalog Oracles** | ✅ 5 adapters | Synthetic, SMNIST, LL, Vision, Code (3 nouveaux squelettes complets) |
| **Catalog Battery** | ✅ + parallel | n_workers > 1 dispatch ThreadPoolExecutor |
| **Catalog Projectors** | ✅ 8 | Toeplitz, Hankel, Cauchy, Banded, Block-diag, Butterfly, Monarch, Pixelfly |
| **Catalog Fast solvers** | ✅ 3 | Levinson, Cauchy, displacement (Kailath) |
| **Sprint runners** | ✅ 10 | B, C, D, E, F, G, S4-S7 — squelettes ou complets selon disponibilité pod |
| **Livrables** | ✅ 6 | cross-Oracle, predictions, signatures, verdict ASP, figures, run_all |
| **Phase 1** | ✅ Complet | Oracle SMNIST entraîné |
| **Phase 1.5** | ✅ Complet | 3 signaux calibrés |
| **Phase 2** | ✅ Code complet | À lancer Sprint C sur dumps |
| **Phase 3** | 🔄 Squelette | Sprint D = backbone informé Sprint C |
| **Phase 4** | 🔄 Squelette | Sprints E + F |
| **Phase 5** | 🔄 Squelette | Sprint G |
| **Tests** | ✅ **719 verts** | + 1 skip OPENBLAS |
| **Doc** | ✅ Tous templates | + 2 outlines paper |
| **Robustesse** | ✅ retry + logs + manifest | Pré-pod ready |

**Bloqueur unique** : exécution réelle des Sprints B-G + S4-S7 sur pod RunPod RTX 5090 (~$50-100, ~2-4 mois wall-clock).

---

## 📖 Guide lecture documentation

1. **Point d'entrée** : [`DOC/INTRODUCTION.md`](DOC/INTRODUCTION.md)
2. **Thèse + vocabulaire** : [`DOC/00_FONDATIONS.md`](DOC/00_FONDATIONS.md)
3. **Catalogue scientifique** : [`DOC/CATALOGUE.md`](DOC/CATALOGUE.md)
4. **Critères falsifiabilité** : [`DOC/falsifiabilite.md`](DOC/falsifiabilite.md)
5. **Phases (01-05)** : [`DOC/01_phase_metrologie.md`](DOC/01_phase_metrologie.md) → ... → [`DOC/05_phase_pareto.md`](DOC/05_phase_pareto.md)
6. **Sprints orchestration** : [`DOC/sprints/README.md`](DOC/sprints/README.md)
7. **Paper outlines** : [`DOC/paper/README.md`](DOC/paper/README.md)
8. **Chronologie + décisions** : [`DOC/carnet_de_bord.md`](DOC/carnet_de_bord.md)
9. **Setup pod** : [`OPS/setup/SETUP.md`](OPS/setup/SETUP.md)

---

**Document généré** : 2026-05-12 fin session | **Pivot stratégique** : Catalogue (Partie 1) prioritaire, ASP (Partie 2) en parallèle conditionné.

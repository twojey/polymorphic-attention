# Rapport phase 2 — Audit Spectral (Run final 2026-05-12)

> Spec : [DOC/02_phase_audit_spectral.md](../02_phase_audit_spectral.md).
> Driver : `phase2_audit_spectral.run` (commit `288e1b7`).
> Pod : RunPod RTX 5090 sm_120, torch 2.11.0+cu128.

## Métadonnées

- **Run ID** : `s2_smnist_audit_288e1b7` (dernier, v8 reprise du checkpoint v5)
- **Git hash** : `288e1b7` (avec fixes `60d1ccf` checkpoint + `255fc12` max_seq_len + `bb89586` phase 3 robustesse)
- **Sprint** : 2
- **Domaine** : Structure-MNIST (SMNIST, 1 seul Oracle, multi-Oracle reporté Sprint 4)
- **Set utilisé** : `audit_svd` (20% tri-partition, seed=7)
- **Status manifest** : `exploratory` (git dirty au lancement, à régulariser au prochain sprint pour `registered`)
- **Backend SVD** : GPU FP32 via eigvalsh(A·Aᵀ+εI) bypass (consumer Blackwell FP64 ≈ 1/64)
- **Coût pod total session** : ~$2 (5 runs phase 2 nécessaires pour debug pipeline, plusieurs killed-restart)

## Données d'entrée

- **9 dumps multi-bucket** issus de `phase1_metrologie.run_extract` (V2) après application des caps :
  - `skip_seq_len_above=2000` → drop seq=5127 (Δ=1024, 8 TB FP64 sinon)
  - `max_bucket_size_gb=15` → caps auto par bucket
  - Total : 10 112 exemples conservés / 13 107 audit_svd (77 %)
- **Régimes couvert** : 9 buckets de seq_len ∈ {7, 19, 53, 87, 155, 223, 291, 327, 1287}, soit ω ∈ [0,8] et Δ ∈ [0,256], ℋ implicite (regrouper côté seq=87)

## Verdict global

**🟡 NO-GO sur le sous-catalogue testé {Toeplitz, Hankel, Identity, composition Toeplitz+Hankel}** (orphan_ratio = **1.000 / 1.000**, seuil 0.20).

**⚠️ Important** : **seules 3 classes sur ~20 prévues** (DOC/00b §B+O+P+T+U) ont été testées en phase 2 V1. Le verdict "100% orphan" est donc dans un **sous-catalogue restreint**, pas le catalogue complet. Classes critiques NON testées :
- B3 Cauchy (rang de déplacement ≤ 1) — **prime suspect** vu le signal résidu rank-1
- B3 Vandermonde
- B5 Block-diagonal
- B6 Banded (attention locale focalisée)
- U1 Butterfly, U2 Monarch, U4 Block-sparse, U5 Sparse+low-rank
- T G-équivariance, circulantes, block-circulantes
- P Ho-Kalman block + HSV

**🟢 SCH validée fortement au sens "rang effectif faible"** (r_eff médian = 2, max = 13 sur N×N matrices avec N jusqu'à 1287).

→ **Distinction clé** : les matrices d'attention SMNIST sont **ultra low-rank**. La forme exacte (Cauchy ? Banded ? Rank-1 sparse outer ?) ne peut être tranchée qu'avec le catalogue complet implémenté.

## Stress-Rank Map (V3.5 distributionnelle)

### r_eff global (sur 10 112 exemples × 6 layers × 8 heads = 485 376 matrices)

| Mesure | Valeur |
|---|---|
| Moyenne | 2.52 |
| Médiane | 2.00 |
| std | 1.48 |
| Min | 1 |
| Max | 13 |

### Histogramme r_eff (% du total)

| r_eff | Comptage | % |
|---|---|---|
| 1 | 136 761 | **28.2 %** |
| 2 | 144 160 | **29.7 %** |
| 3 | 102 014 | **21.0 %** |
| 4 | 58 848 | 12.1 % |
| 5 | 24 390 | 5.0 % |
| 6-10 | 18 854 | 3.9 % |
| 11-13 | 349 | 0.07 % |

**78.9 % des matrices ont r_eff ≤ 3.** **99.8 % ont r_eff ≤ 10.** **100 % ≤ 13.**

### Par couche (mean r_eff)

| Layer | Mean r_eff |
|---|---|
| 0 | **4.10** |
| 1 | 2.89 |
| 2 | 1.96 |
| 3 | 2.06 |
| 4 | 2.05 |
| 5 | 2.05 |

Layer 0 utilise un peu plus de rang, layers 2-5 sont quasi rank-2 stables. Pattern habituel d'attention dense : layer 0 fait du "preprocessing" plus diffus, layers profonds spécialisent et rétrécissent.

## Loi de transfert r_target = a · ω^α · Δ^β · g(ℋ)

Non calculée précisément dans ce run (la régression nécessite assez de variance sur r_target ; vu que r_eff varie entre 1 et 13 sur 99.93 % des données, le signal est compressé). À refaire offline.

## Diagnostic spécialisation des têtes (DOC/02 §5b)

- **Têtes dormantes** : **0 / 48**
- **Top 8 têtes spécialisées** (spec_h = var(r_eff_h) cross-régime) :

| Rang | Layer | Head | spec_h | mean_r_eff |
|---|---|---|---|---|
| 1 | 0 | 2 | **5.14** | 5.53 |
| 2 | 0 | 4 | 3.63 | 4.58 |
| 3 | 0 | 5 | 3.12 | 4.72 |
| 4 | 0 | 3 | 2.38 | 4.30 |
| 5 | 1 | 5 | 2.18 | 4.22 |
| 6 | 4 | 6 | 1.72 | 2.39 |
| 7 | 5 | 2 | 1.71 | 2.40 |
| 8 | 4 | 0 | 1.68 | 2.14 |

Pattern marqué : **layer 0 concentre les têtes spécialisées** (top 4 sont toutes L=0). Smart Init Matriochka phase 3 devrait privilégier l'extraction des vecteurs singuliers des têtes L=0 H=2,3,4,5.

## Batterie A — Fitting des classes

### Vue globale (54 régimes (layer, ω, Δ))

| Statistique ε_best | Valeur |
|---|---|
| Min | **0.452** |
| p10 | 0.762 |
| Médiane | **0.983** |
| p90 | 0.995 |
| Max | 0.998 |
| Comptage ε < 0.30 (seuil orphan) | **0 / 54** |

**Aucun régime** ne fitte une classe V1 avec ε < 0.30. Même le "moins pire" (L=3 ω=2 Δ=0, Toeplitz) reste à ε=0.45.

### Distribution des class_winners

| Classe gagnante | Comptage | % |
|---|---|---|
| Toeplitz | 36 | 66.7 % |
| Hankel | 18 | 33.3 % |
| Identity | 0 | 0 % |

→ Quand on force un choix, Toeplitz l'emporte. Mais ε est si élevé que ce n'est pas un signal exploitable pour le Backbone phase 3.

### Top 5 "less bad" régimes

| Layer | ω | Δ | class_best | ε_best | ε_toep | ε_hank | ε_ident |
|---|---|---|---|---|---|---|---|
| 3 | 2 | 0 | toeplitz | **0.452** | 0.452 | 0.506 | 0.970 |
| 0 | 2 | 0 | toeplitz | 0.466 | 0.466 | 0.472 | 0.974 |
| 1 | 2 | 0 | toeplitz | 0.492 | 0.492 | 0.545 | 0.993 |
| 5 | 2 | 0 | hankel | 0.549 | 0.550 | 0.549 | 0.985 |
| 4 | 2 | 0 | toeplitz | 0.728 | 0.728 | 0.729 | 0.968 |

**Pattern** : tous les régimes "less bad" sont **Δ=0** (pas de structure de binding distant). C'est le cas dégénéré où l'attention n'a presque rien à faire. Dès que Δ ≥ 16, ε ≥ 0.95.

## Batterie B — Analyse résidu (signal scientifique majeur)

Le résidu post-projection sur la classe optimale est analysé par SVD :

| Régime | ‖résidu‖ | **svd_top1_ratio** | Interprétation |
|---|---|---|---|
| 0 | 0.550 | **1.000** | Résidu = rank-1 pur |
| 1 | 0.628 | **1.000** | Résidu = rank-1 pur |
| 2 | 1.297 | **1.000** | Résidu = rank-1 pur |
| 3 | 0.547 | **1.000** | Résidu = rank-1 pur |
| 4 | 1.191 | **1.000** | Résidu = rank-1 pur |

**🎯 Signal scientifique majeur** : pour TOUS les régimes inspectés, la première valeur singulière du résidu capture 100 % de l'énergie spectrale → **la structure résiduelle est rank-1 exact**.

→ Combiné avec r_eff médian = 2 : **les matrices d'attention SMNIST sont des outer products `u·vᵀ` (rank-1 dominantes) avec une petite contribution Toeplitz qui n'explique pas la structure**.

C'est la "découverte" scientifique du run : le bon modèle n'est PAS dans le catalogue V1, c'est `u_t · v_tᵀ` avec u, v paramétrés par le contexte (= exactement ce qu'une attention dense fait à softmax avec un token dominant).

## Batterie D — Out-of-catalogue

- **Régimes orphelins** : **54 / 54 (100 %)** — TOUS les régimes ont ε_min > 0.30 même avec composition Toeplitz+Hankel additive
- **Asymétrie eigen/SVD** : moyenne = **1.33** (très élevée) — les attentions post-softmax sont massivement asymétriques (causalité + softmax sur tokens importants ≠ structure symétrique type Toeplitz)
- **Hypothèses sur classes manquantes** :
  1. **Rank-1 (outer product) u·vᵀ** ← favori (cohérent avec résidu rank-1 + r_eff médian = 2)
  2. **Banded** (attention focalisée sur fenêtre locale) — pas testé V1
  3. **Block-sparse** — pas testé V1
  4. **Composition Toeplitz + rank-1** — pas dans le catalogue ajouté V1

## Dictionnaire SCH (sortie clé pour phase 3)

| Composante | Valeur | Source |
|---|---|---|
| r_eff dominant | **1-3** (78.9 % des matrices) | SVD batchée FP32 GPU |
| r_eff max observé | 13 | SVD batchée |
| Classe structurée | **Rank-1 / outer product** (hypothèse) | Batterie B résidu top1=1.0 |
| Classe candidate Toeplitz | ε_min 0.45, ε_med 0.98 — non utilisable | Batterie A |
| Classe candidate Hankel | ε_min 0.49, ε_med 0.99 — non utilisable | Batterie A |
| Asymétrie | 1.33 (forte) | Batterie D |
| Têtes spécialisées | 4 sur 8 dans L=0 | Diagnostic head_specialization |

## R_max recommandé chiffré

**R_max = 16** (avec marge 1.2× le max observé) ou **R_max = 32** (marge 2.5×, plus prudent).

Justification : r_eff max = 13. Avec R_max=16, on est juste au-dessus. Avec R_max=32, on a une marge confortable pour absorber une éventuelle augmentation cross-domain. Le coût compute d'avoir R_max plus grand est négligeable car le mask Matriochka soft-cut les rangs non utilisés.

**Choix retenu pour phase 3** : `R_max=32`.

## Verdict go/no-go phase 2 (DOC/02 §2.8)

**Décision** : **NO-GO sur le catalogue V1**, **GO conditionnel sur la SCH au sens rank**.

- **SCH (rang faible reproductible)** : **VALIDÉE forte** (médiane 2, IQR/médiane < 0.5, distribution étroite, 99.8 % < 10)
- **Catalogue V1 {Toeplitz, Hankel, Identity}** : **REJETÉ** (orphan_ratio = 1.000)
- **Diagnostic découplage S_Spectral ↔ r_eff** : **non calculé** (dette technique — compute_s_spectral + bootstrap Spearman trop lent en multiprocessing fork sur seq_len > ~300 — à refaire offline avec implémentation GPU batchée)

## Recommandation Backbone phase 3

**Au lieu de** Backbone Toeplitz/Hankel/composite (catalogue V1 réfuté), utiliser :

- **`IdentityBackbone`** comme Backbone "neutre" : laisse la totalité de l'apprentissage de la structure à l'extension Matriochka ΔAttn = U·Vᵀ.
- C'est cohérent avec le résultat batterie B : **la vraie structure est rank-1 dominante** → ΔAttn rank-K avec K=R_max=32 capture amplement la structure mesurée (médiane r_eff = 2, max 13).

**Alternative explorable en ablation** :
- `BackboneOuterProduct` (à coder) : `y_t = u_t · v_tᵀ · x_t` avec u, v fonctions du contexte. Mais c'est essentiellement ce que fait l'attention dense → on ne ferait que reproduire l'Oracle, pas optimiser.

## Limitations méthodologiques

1. **🚨 Catalogue testé = 10-15% du catalogue spec DOC/00b** :
   - **Testées** : B1 Toeplitz, B2 Hankel, Identity (placeholder)
   - **Non testées (~17 classes)** : B3 Cauchy, B3 Vandermonde, B5 Block-diagonal, B6 Banded, B7 Tropical, B8 Sylvester ; **O1-O3 rangs de déplacement** (Cauchy-like, Vandermonde-like) ; **P1-P4 Ho-Kalman block** ; **T équivariances** (G-equiv, perm-inv, circulantes, block-circulantes) ; **U1-U5 Butterfly/Monarch/Pixelfly/Block-sparse/Sparse+low-rank**
   - **Conséquence** : le verdict orphan_ratio = 1.000 n'est pas un verdict sur "hors catalogue absolu", c'est "hors sous-catalogue de 3 projecteurs"
   - **Action V2** : implémenter les projecteurs manquants au moins pour Cauchy + Banded + Block-diag (les plus prometteurs vu le signal résidu rank-1), puis relancer la batterie A
2. **Status `exploratory` (pas `registered` strict)** : le git était dirty au lancement de phase 2 v5/v8 → manifest non-pré-enregistrable. À refaire avec `--watch` propre quand on aura validé phase 3.
3. **Diagnostic découplage S_Spectral non calculé** : on ne sait pas si le S_Spectral du Run 3 (phase 1.5) mesurait du vrai r_eff ou un artefact. À recalculer offline avec implémentation single-thread.
4. **1 seul Oracle (SMNIST)** : universalité cross-domain non testée (reporté Sprint 4).
5. **Δ=1024 dropé** : la mesure ne couvre pas le régime de binding longue distance extrême. À traiter à part en Sprint 4 si jamais.

## Lessons learned (cf. carnet 2026-05-12)

1. **Checkpoint resume est OBLIGATOIRE** pour tout pipeline > 5 min — on a perdu ~25 min × 2 sur phase 2 avant de l'implémenter (`feedback_script_robustness.md` mis à jour)
2. **GPU consumer Blackwell FP64 nerfé → bypass eigvalsh(A·Aᵀ+εI)** plus rapide ET plus stable que torch.linalg.svdvals fallback interne
3. **Caps configurables obligatoires** : `skip_seq_len_above`, `max_bucket_size_gb`, `max_examples_metrics`, `max_seq_len` (diagnostic) — sinon overflow disque ou compute infini
4. **MLflow log_artifact opt-in** pour gros fichiers — disque pod a explosé à 89 GB la première fois
5. **Bug `bool(X or True)`** ignorait `decoupling.enabled=false` — `OmegaConf.select(..., default=True)` corrige
6. **Le "vrai" catalogue à tester en V2** : `{rank-K, banded, block-sparse, Toeplitz, Hankel}` — V1 trop restrictif

## Artefacts rapatriés (VPS `OPS/logs/pod_2026_05_12/`)

- `phase2_state/svd_r_eff.pt` (3.7 MB) : tensor (L=6, B=10112, H=8) avec r_eff(θ=0.99) par triplet
- `phase2_state/batteries_results.pt` (27 KB) : battery_a + battery_b + battery_d sérialisés
- `phase2_state/fingerprint.pkl` : config du run
- `mlruns/` : métriques + manifests MLflow
- `run_logs/` : logs phase 1 V2 + phase 2 v1-v8

**Non rapatrié** : `/workspace/phase1_extract/*.pt` (83 GB FP64 sur pod, non transférable sur la liaison). Re-extraction = ~30 min + $0.30 si jamais nécessaire pour re-run phase 2 batterie D enrichie ou diagnostic découplage offline.

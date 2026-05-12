# ROADMAP — projet ASP

> **État** : 2026-05-12 fin session — **scaffolding complet, ready-to-pod**.
>
> **Pivot stratégique** : Partie 1 (Catalogue) prioritaire sur Partie 2 (ASP). Voir DOC/CATALOGUE.md.
>
> **Bloqueur unique** : exécution réelle sur pod RunPod RTX 5090 (~$50-100, ~2-4 mois wall-clock total).

---

## 🎯 État du code (fin de session)

| Bloc | État | Détails |
|---|---|---|
| **Properties Catalog** | ✅ **131 / 131 (100 %)** | 23 familles A-W + N, catalogue complet |
| **Oracles adapters** | ✅ 5 | Synthetic, SMNIST + 3 nouveaux complets (LL, Vision, Code) |
| **Battery** | ✅ + parallèle | `n_workers > 1` ThreadPoolExecutor |
| **Projectors** | ✅ 8 | Toeplitz, Hankel, Cauchy, Banded, Block-diag, Butterfly, Monarch, Pixelfly |
| **Fast solvers** | ✅ 3 | Levinson-Durbin, Cauchy, Sylvester displacement |
| **Sprint runners** | ✅ 10 | B, C, **D, E, F, G** (wirés via subprocess), S4-S7 |
| **Livrables génération** | ✅ 6 | cross-Oracle, predictions, signatures, verdict ASP, figures, run_all |
| **Robustesse** | ✅ | retry helper, logs UTC horodatés, manifest reproductible |
| **Phase 1** | ✅ Oracle entraîné | `oracle_e2f0b5e.ckpt` |
| **Phase 1.5** | ✅ Calibré | 3 signaux validés |
| **Phase 2-5** | ✅ **Drivers complets** | phase4 inner loops + phase5 5 tests implémentés |
| **Paper outlines** | ✅ 2 | Partie 1 + Partie 2 |
| **Tests** | ✅ **724 verts** | + 1 skip OPENBLAS |

---

## 🚀 Prochaines actions

### Hors pod (faisables VPS)

Tout codable est déjà fait. Restent uniquement :
1. **Rédaction paper Partie 1** post-Sprint C (humain ~3-4 sem)
2. **Lecture critique** du catalog DOC/CATALOGUE.md (review math)

### Sur pod (à exécuter)

> **Pod-split obligatoire** : Sprints B/C/S4 sur **pod CPU 251 GB RAM** ($0.10-0.20/h), Sprints D-G sur **pod GPU RTX 5090** ($0.50-0.80/h). Cf. section "Calendrier compute" ci-dessous.

```bash
# Bootstrap pod CPU (pour B, C, S4) — Sprint B exige RAM > VRAM
ssh -p $POD_CPU_PORT root@$POD_CPU_IP "cd /workspace/polymorphic-attention && \
    bash OPS/scripts/setup_pod.sh"

# Sprint B (re-extraction dumps phase 1 V2) — 15-20 min CPU, $0.05-0.07
bash OPS/setup/launch_sprint.sh B --nohup --watch --device cpu -- \
    --oracle-checkpoint OPS/checkpoints/oracle_e2f0b5e.ckpt

# Sprint C (Battery research × 131 props × 9 dumps) — CPU + GPU hybride si dispo
# ~80 % props FP64-SVD bloquées CPU, ~15-25 % GPU-affines (FFT/cdist) si pod a GPU
bash OPS/setup/launch_sprint.sh C --nohup --device cpu -- \
    --dumps-dir OPS/logs/sprints/B_re_extract/dumps

# Génération livrables Partie 1
PYTHONPATH=CODE uv run python -m livrables.run_all \
    --results OPS/logs/sprints/C_catalog_full/results.json:OR \
    --predictions DOC/paper/partie1/predictions_a_priori.yaml \
    --output DOC/paper/partie1/

# Sprints S5-S7 (Oracles cross-domain, parallèle Sprint C+D)
bash OPS/setup/launch_sprint.sh S5 --device cuda --nohup -- --use-hf-dinov2
bash OPS/setup/launch_sprint.sh S7 --device cuda --nohup -- --use-hf-llama

# Sprint D (phase 3 V3+ avec Backbone informé Sprint C) — 1-2 sem, $5-10
bash OPS/setup/launch_sprint.sh D --device cuda --nohup -- \
    --sprint-c-report DOC/reports/sprints/sprint_c.md

# Sprints E-F-G (phase 4-5 ASP) — 1 sem, $30-50
bash OPS/setup/launch_sprint.sh E --device cuda --nohup --
bash OPS/setup/launch_sprint.sh F --device cuda --nohup --
bash OPS/setup/launch_sprint.sh G --device cuda --nohup --

# Verdict ASP final
PYTHONPATH=CODE uv run python -m livrables.partie2_asp_verdict \
    --test-results OPS/logs/sprints/G_phase5_validation/test_results.json \
    --output DOC/paper/partie2/
```

---

## 🖥️ Calendrier compute (CPU vs GPU par tâche)

**Règle d'allocation** : un sprint/property va sur GPU si **(a)** speedup mesuré ≥ 3× ET **(b)** fit en VRAM 32 GB (5090) ou cible plus large dispo. Sinon CPU.

**Contraintes structurelles** :

- **Blackwell sm_120 limité FP64** : ~40 properties marquées `requires_fp64=True` (familles O, P, Q surtout) sont **CPU-obligatoires** sans alternative. Inutile de tenter GPU.
- **VRAM 32 GB hard cap** sur la 5090 : régimes Sprint B `ω≥2 Δ=256` demandent 80-150 GB VRAM (cf. carnet 2026-05-12 OOM observé : 81.78 GiB requis sur 31.37 GiB dispo). Aucun GPU consumer/cloud raisonnable ne tient (5090, A100, H200 tous OOM sur ω=4 Δ=256 ~150 GB). Seul **CPU 251 GB RAM** ou **B200 192 GB** ($5-7/h, indispo Community) tiennent.

### Taxonomie par sprint

| Sprint | Charge dominante | Pod | $/h | Justification |
|---|---|---|---:|---|
| **B** Re-extract dumps SMNIST | Forward Oracle + save RAM-bound | CPU | $0.10-0.20 | Régimes ω=4 Δ=256 demandent ~150 GB ; seul CPU 251 GB RAM tient les 9/9 régimes |
| **C** Catalog 131 props × dumps | ~80 % SVD FP64 + ~20 % FFT/cdist | CPU (+GPU opt.) | $0.10-0.20 | FP64 CPU-bound. Si pod CPU+GPU dispo, dispatch hybride post-bench |
| **S4** SMNIST seq 1024-4096 | Forward étendu, RAM-bound | CPU | $0.10-0.20 | Plus gros que B, OOM VRAM garanti |
| **S5** Vision DINOv2 (87M) | HF inference transformers | GPU léger | $0.30-0.50 | A priori GPU 10-100× vs CPU sur forward HF, à valider |
| **S6** Code StarCoder-2-3B | HF inference | GPU léger | $0.30-0.50 | Idem S5 |
| **S7** LL GPT-2 + Llama-3.2-1B | HF inference | GPU léger | $0.30-0.50 | Idem |
| **D** ASPLayer training 50ep | Training NN avec ASP kernel | **GPU 5090 Blackwell** | $0.50-0.80 | ASP kernel exige sm_120 (cf. STACK.md) |
| **E**-**F**-**G** Phase 4-5 ASP | Training NN + validation | **GPU 5090** | $0.50-0.80 | Idem D |

### Properties Sprint C : affinité par famille (inventaire statique 2026-05-12)

| Famille | Props | Pattern dominant | Affinité GPU |
|---|---:|---|---|
| O, P, Q (déplacement, réalisation, hiérarchique) | ~18 | SVD FP64 obligatoire | 🔴 CPU only |
| S, K (tenseurs, graph) | ~10 | SVD + einsum, partiel FP64 | 🔴 CPU majoritaire |
| B (structurelles) | 10 | SVD léger + distances | 🟡 dépend taille |
| C (token stats) | 9 | Entropies/scalaires | 🟡 trop petit, CPU OK |
| L (fréquentiel) | 5 | 3× `torch.fft` + einsum | 🟢 GPU candidat fort |
| R (RKHS) | 6 | 1 FFT + 2 SVD FP32 | 🟢 GPU candidat |
| V (opérateurs) | 5 | 1 FFT | 🟢 GPU candidat |
| W (logique) | 5 | 1 `torch.cdist` (W2 fix 2026-05-12) | 🟢 GPU candidat |
| G, I, M, T, U, autres | ~25 | scalaires + petits ops | 🟡 à bencher |

**Bilan estimé** : ~17-25 props CPU-obligatoires, ~15-20 GPU-candidates fortes, ~80 grey zone. Même un Sprint C "tout-GPU" n'accélère que ~15-25 % du travail.

### Benchmark CPU vs GPU — résultats empiriques 2026-05-12

Script : `OPS/scripts/bench_cpu_vs_gpu_properties.py` (11 properties × 3 régimes × CPU/GPU, médiane 3 reps).

**Run pod RTX 5090 sm_120 / PyTorch 2.11.0+cu128 — 4 samples par dump, layer 0** : `OPS/logs/benchmarks/bench_cpu_vs_gpu_20260512T195455Z.md`.

**Speedups CPU/GPU mesurés** (≥3× = GPU recommandé) :

| Property | Δ=16 (N=19) | Δ=64 (N=67) | Δ=256 (N=259) |
|---|---:|---:|---:|
| `L1_fft2d_energy` (FFT 2D) | 0.70× | 1.82× | **19.56× 🟢** |
| `L3_quasi_periodicity` (FFT 1D) | 0.74× | 1.60× | **21.53× 🟢** |
| `W2_dependence_proxy` (cdist post-fix) | 1.03× | 2.10× | **8.50× 🟢** |
| `D1_head_cosine` (cdist/cosine) | 0.56× | 0.68× | 2.15× |
| `B8_sylvester_rank` (SVD FP64) | 0.31× | 0.76× | 0.78× |
| `O1_toeplitz_displacement_rank` (SVD FP64) | 0.57× | 0.90× | 2.62× |
| `B1_toeplitz_distance` (scalaire) | 0.48× | 0.65× | 1.30× |
| `B4_sparse_fraction` (scalaire) | 0.64× | 1.46× | **10.92× 🟢** |
| `C3_shannon_entropy` (scalaire) | 0.66× | 1.13× | 2.68× |
| `I1_head_diversity` (einsum) | 0.88× | **4.43× 🟢** | **68.41× 🟢** |
| `R3_bochner_stationarity` (FFT + SVD) | 0.41× | 0.47× | 1.18× |

**Interprétations** :

1. **Δ=16 (N=19) → tout CPU**. Overhead PCIe domine sur petites matrices, aucun speedup ≥ 1×.
2. **Δ=64 (N=67) → frontière**. Seul `I1_head_diversity` (einsum) franchit le 3× (4.43×). Le reste reste en CPU.
3. **Δ=256 (N=259) → GPU gagne sur 5 properties** (FFT, cdist, einsum, sparse_fraction) avec speedups massifs (8-68×). SVD FP64 reste CPU partout (Blackwell FP64 limité confirmé : B8 0.31× sur petit, 0.78× sur grand).
4. **Surprise : `B4_sparse_fraction` (10.92× à Δ=256)** — boolean masking sur 259×259, GPU bat CPU largement.
5. **Caveat** : bench sur 4 samples seulement. En production (B=512), les ratios peuvent évoluer (overhead GPU mieux amorti → encore plus favorable, ou bandwidth-limited → moins).

**Recommandation device par sprint** :

- **Sprint B** (extraction) : CPU obligatoire (VRAM hard cap).
- **Sprint C tout-CPU** : OK, simple. Acceptable parce que ~80 % des 131 properties sont CPU-bound de toute façon (FP64 SVD).
- **Sprint C hybride CPU+GPU** : gain potentiel ~10-25 % wall-clock SI dispatcher Battery sait acheminer (`L*`, `I1`, `W2`, `B4`) vers GPU sur régimes Δ≥256, le reste CPU. Nécessite patch ~50 LoC dans `catalog/batteries/base.py:_process_one_regime`. Pas urgent — à évaluer si Sprint C est trop lent.
- **Sprint S5-S7** : non bencheable ici (forward HF transformers), benchmark dédié à faire sur pod cible.

```bash
PYTHONPATH=CODE uv run python OPS/scripts/bench_cpu_vs_gpu_properties.py \
    --dumps-dir OPS/logs/sprints/B/dumps --repeats 3
```

### Stratégie globale

- **Maintenant → Partie 1 finie** : pod CPU dédié pour B + C + S4. 5090 actuelle à killer dès que Sprint C en cours sort (run baseline 6/9 dumps).
- **Sprints S5-S7** : choix pod CPU ou GPU léger selon bench HF (`transformers` peut tourner CPU mais 10-100× plus lent).
- **Quand Sprint D démarre** : relouer pod 5090 (re-setup 5-10 min via `setup_pod.sh` idempotent, checkpoint Oracle versionné côté VPS).

Économie estimée pod-split vs "5090 active tout du long" : $0.50/h × 24h × 3-7 j Partie 1 = **$36-84**.

---

## 📦 Sprints en détail

### Sprint B — Re-extraction phase 1 V2
- **Objectif** : 9 dumps multi-bucket (ω × Δ) sur SMNIST Oracle entraîné
- **Pré-requis** : checkpoint Oracle SMNIST (✅)
- **⚠ Device obligatoire** : `--device cpu`. Sweep 3×3 produit 3 régimes OOM VRAM sur 5090 (ω=2 Δ=256 demande 82 GB, ω=4 Δ=256 ~150 GB). Le run GPU 2026-05-12 a produit 6/9 dumps puis abandonné les 3 lourds. Spec `OPS/configs/sprints/sprint_b.yaml:11` met `device: cpu` par défaut, ne pas override.
- **Compute** : 15-20 min pod CPU, ~$0.05-0.07 ($0.30 estimation initiale était basée sur GPU partiel)
- **Output** : `OPS/logs/sprints/B_re_extract/dumps/dump_omegaX_deltaY.pt` × 9
- **Tailles dumps observées** (B=512, 6 layers, 8 heads, FP32) :
  - ω=0 : Δ=16 → 68 MB, Δ=64 → 842 MB, Δ=256 → 13 GB
  - ω=2 : Δ=16 → 1.4 GB, Δ=64 → 20 GB, Δ=256 → ~60 GB (estim.)
  - ω=4 : Δ=16 → 4.4 GB, Δ=64 → ~50 GB (estim.), Δ=256 → ~150 GB (estim.)
  - Croissance ~24× ω=0→ω=2 à Δ=64 fixé — vérifier mécanisme exact lors du run CPU
- **Bloquant pour** : Sprint C (9 dumps requis pour M2 `scope=cross_regime`)
- **Code** : `CODE/sprints/sprint_b_re_extract.py` — retry sur extract_regime, checkpoint/resume par régime
- **Template rapport** : `DOC/reports/sprints/sprint_b_template.md`

### Sprint C — Battery research × dumps
- **Objectif** : 131 Properties sur 9 dumps → signatures complètes SMNIST
- **Pré-requis** : Sprint B
- **Compute** : ~1 sem CPU, $5-10
- **Output** : `results.json` + `report.md` Markdown
- **Critères go** : ≥ 50 % des Properties produisent une valeur ; au moins 1 classe identifiée
- **Code** : `CODE/sprints/sprint_c_catalog_full.py` + `_FrozenDumpOracle`
- **Template rapport** : `DOC/reports/sprints/sprint_c_template.md`

### Sprint D — Phase 3 V3+ Backbone informé
- **Objectif** : ASPLayer avec projector identifié Sprint C
- **Pré-requis** : Sprint C report
- **Compute** : 1-2 sem GPU pod, $5-10
- **Critères go** : val_acc ASP ≥ 0.90 × Oracle baseline (0.580 ≥ 0.580)
- **Code** : `CODE/sprints/sprint_d_phase3_v3.py` — parse Sprint C report, sélectionne backbone

### Sprints E + F — Phase 4 warm-up + autonomous
- **Sprint E** : Spectromètre avec ground-truth ω/Δ/ℋ (1 jour, $5-10)
- **Sprint F** : Routing autonome sans signaux ground-truth (2 jours, $10-20)
- **Critère F** : val_acc ≥ 0.90 × Oracle SANS signaux ground-truth (test crucial)

### Sprint G — Phase 5 validation finale
- **Tests 5a (identifiabilité), 5b (élasticité), 5c (SE/HT), 5d (anti-fraud), 5e (OOD), 6c (R_max/2)**
- **Verdict ASP** : mandatory 5a + 5c + 6c (autres bonus)
- **Compute** : 3 jours pod, $15-25
- **Code** : `CODE/sprints/sprint_g_phase5_validation.py`

### Sprints S4-S7 — Oracles cross-domain
- **S4** : SMNIST seq_len étendu 1024-4096
- **S5** : Vision DINOv2 (HF) ou MinimalViT
- **S6** : Code StarCoder (HF) ou MinimalCode + Dyck-k
- **S7** : LL Llama-3.2-1B (HF) ou MinimalLM + TinyStories
- **Cible** : signatures cross-Oracle pour paper Partie 1

---

## 📚 Décisions cadres

| # | Décision | Détails | Source |
|---|---|---|---|
| 0.1 | **Stack** : PyTorch 2.11+cu128 + Fabric + Hydra + uv | sm_120 Blackwell | OPS/env/STACK.md |
| 0.3 | **Hardware** : VPS + RunPod éphémère, **pod-split CPU/GPU** | CPU 251 GB pour B/C/S4, RTX 5090 sm_120 pour D-G. Règle : GPU si speedup ≥3× ET fit VRAM | OPS/env/HARDWARE.md + section "Calendrier compute" |
| 0.4 | **Logging** : MLflow self-hosted sur VPS | tunnel SSH inverse | OPS/env/LOGGING.md |
| — | **Pivot 2026-05-12** | Catalogue (Partie 1) prioritaire sur ASP | DOC/CATALOGUE.md + carnet |

---

## 📖 Documentation prioritaire

1. **Point d'entrée** : [`DOC/INTRODUCTION.md`](DOC/INTRODUCTION.md)
2. **Thèse + vocabulaire** : [`DOC/00_FONDATIONS.md`](DOC/00_FONDATIONS.md)
3. **Catalogue scientifique** : [`DOC/CATALOGUE.md`](DOC/CATALOGUE.md) (livrable Partie 1)
4. **Sprints orchestration** : [`DOC/sprints/README.md`](DOC/sprints/README.md)
5. **Paper outlines** : [`DOC/paper/README.md`](DOC/paper/README.md)
6. **Falsifiabilité** : [`DOC/falsifiabilite.md`](DOC/falsifiabilite.md)
7. **Chronologie** : [`DOC/carnet_de_bord.md`](DOC/carnet_de_bord.md)
8. **Setup pod** : [`OPS/setup/SETUP.md`](OPS/setup/SETUP.md)

---

## 🧪 Garanties robustesse pré-pod

Toutes implémentées et testées :
- **Logs horodatés UTC** par Sprint (`<output_dir>/sprint.log`)
- **Checkpoint atomique** + resume après crash (fingerprint mismatch → RuntimeError explicite)
- **Retry transients** : `shared/retry.py` (backoff exp + jitter, KI/SystemExit jamais retry)
- **Manifest reproductible** : git hash + dirty flag, torch version, python, cuda
- **MLflow opt-in** (graceful si URI absent)
- **Aucune erreur silencieuse** : tous les `except` loguent (logger.exception)
- **Critères go/no-go explicites** : tracés dans summary.json + log
- **Bootstrap pod** : `launch_sprint.sh` (nohup + watch + ASP_LOG_FILE persisté)

---

## 📈 Historique compact (commits récents)

| Commit | Description |
|---|---|
| `b973488` | Robustesse production-ready (retry + logs + manifest + run_all) |
| `d9d00d8` | Battery dispatch parallèle régimes (n_workers > 1) |
| `fbcdae1` | Scaffolding sprints + livrables + paper outlines + configs |
| `25d943b` | Oracles LL/Vision/Code complets + fast solvers + gudhi opt + cache SVD |
| `91d19a4` | Vague V2 frontière 22 props (98 props total) |
| `4225f50` | Drivers phase 4/5 + cross-oracle harness |
| `a4204ad` | Vague complétion V1 16 props (76 → vague 1 complete) |

Voir `git log --oneline` pour l'historique complet (50+ commits cette session).

---

## ⏳ Estimation effort restant total

| Bloc | Wall-clock | Compute |
|---|---|---|
| Sprint B (re-extract, CPU 9/9 régimes) | 15-20 min | $0.05-0.07 |
| Sprint C (catalog 131 props × 9 dumps, CPU dominant) | 1 sem (à raffiner post-bench) | $5-10 |
| Sprints S4-S7 (cross-Oracle) | 1-2 sem | $20-30 |
| Sprint D (phase 3 V3+) | 1-2 sem | $5-10 |
| Sprints E-F-G (phase 4-5 ASP) | 1 sem | $30-50 |
| Paper Partie 1 rédaction | 3-4 sem | $0 (humain) |
| Paper Partie 2 rédaction | 2-3 sem | $0 (humain) |
| **Total (Partie 1 + verdict Partie 2)** | **~3-4 mois** | **~$60-100** |

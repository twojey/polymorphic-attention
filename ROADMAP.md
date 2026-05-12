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
| **Sprint runners** | ✅ 10 | B, C, D, E, F, G, S4-S7 (squelettes structurés) |
| **Livrables génération** | ✅ 6 | cross-Oracle, predictions, signatures, verdict ASP, figures, run_all |
| **Robustesse** | ✅ | retry helper, logs UTC horodatés, manifest reproductible |
| **Phase 1** | ✅ Oracle entraîné | `oracle_e2f0b5e.ckpt` |
| **Phase 1.5** | ✅ Calibré | 3 signaux validés |
| **Phase 2-5** | 🔄 Squelettes | Code complet, run = Sprint B-G sur pod |
| **Paper outlines** | ✅ 2 | Partie 1 + Partie 2 |
| **Tests** | ✅ **719 verts** | + 1 skip OPENBLAS |

---

## 🚀 Prochaines actions

### Hors pod (faisables VPS)

Tout codable est déjà fait. Restent uniquement :
1. **Rédaction paper Partie 1** post-Sprint C (humain ~3-4 sem)
2. **Lecture critique** du catalog DOC/CATALOGUE.md (review math)

### Sur pod (à exécuter)

```bash
# Bootstrap pod (RTX 5090 pour Sprints D-G, CPU pour B-C)
ssh -p $POD_PORT root@$POD_IP "cd /workspace/polymorphic-attention && \
    bash OPS/scripts/setup_pod.sh"  # ou setup_pod_cpu.sh selon usage

# Sprint B (re-extraction dumps phase 1 V2) — 30 min, $0.30
bash OPS/setup/launch_sprint.sh B --nohup --watch -- \
    --oracle-checkpoint OPS/checkpoints/oracle_e2f0b5e.ckpt

# Sprint C (Battery research × dumps) — 1 sem CPU, $5-10
bash OPS/setup/launch_sprint.sh C --nohup -- \
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

## 📦 Sprints en détail

### Sprint B — Re-extraction phase 1 V2
- **Objectif** : 9 dumps multi-bucket (ω × Δ) sur SMNIST Oracle entraîné
- **Pré-requis** : checkpoint Oracle SMNIST (✅)
- **Compute** : 30 min pod CPU, ~$0.30
- **Output** : `OPS/logs/sprints/B_re_extract/dumps/dump_omegaX_deltaY.pt`
- **Bloquant pour** : Sprint C
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
| 0.3 | **Hardware** : VPS + RunPod RTX 5090 éphémère | training pod, lecture VPS | OPS/env/HARDWARE.md |
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
| Sprint B (re-extract) | 30 min | $0.30 |
| Sprint C (catalog complet) | 1 sem | $5-10 |
| Sprints S4-S7 (cross-Oracle) | 1-2 sem | $20-30 |
| Sprint D (phase 3 V3+) | 1-2 sem | $5-10 |
| Sprints E-F-G (phase 4-5 ASP) | 1 sem | $30-50 |
| Paper Partie 1 rédaction | 3-4 sem | $0 (humain) |
| Paper Partie 2 rédaction | 2-3 sem | $0 (humain) |
| **Total (Partie 1 + verdict Partie 2)** | **~3-4 mois** | **~$60-100** |

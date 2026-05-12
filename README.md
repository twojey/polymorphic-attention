# Attention Superlinéaire Polymorphe (ASP)

**Projet d'étude des propriétés mathématiques de l'attention et d'optimisation sub-quadratique.**

---

## 🚀 Démarrage rapide

**Lire d'abord** : [`DOC/INTRODUCTION.md`](DOC/INTRODUCTION.md) — point d'entrée unique (5 min)

Ensuite selon votre besoin :
- **Comprendre la thèse** → [`DOC/00_FONDATIONS.md`](DOC/00_FONDATIONS.md) (la science en 30 min)
- **Voir le catalogue complet** → [`DOC/CATALOGUE.md`](DOC/CATALOGUE.md) (131 propriétés)
- **Connaître l'état courant** → [`ROADMAP.md`](ROADMAP.md) (état + prochaines actions)
- **Entendre la chronologie** → [`DOC/carnet_de_bord.md`](DOC/carnet_de_bord.md) (journal vivant)
- **Comprendre la structure code** → [`ARBORESCENCE.md`](ARBORESCENCE.md) (CODE/DOC/OPS)

---

## 📚 Structure documentation

Tous les documents sont consolidés en **11 fichiers principaux** :

### **Fondations (3 fichiers)**
- `DOC/INTRODUCTION.md` — Point d'entrée, table des matières
- `DOC/00_FONDATIONS.md` — Thèse ASP + vocabulaire architectural + mathématique
- `DOC/CATALOGUE.md` — 131 propriétés × 6 Oracles × prédictions a priori

### **Phases scientifiques (6 fichiers)**
- `DOC/01_phase_metrologie.md` — Phase 1 : Métrologie (RCP/SSG)
- `DOC/01b_phase_calibration_signal.md` — Phase 1.5 : Identification Gate
- `DOC/02_phase_audit_spectral.md` — Phase 2 : Audit spectral (SCH)
- `DOC/03_phase_kernel_asp.md` — Phase 3 : ASPLayer (architecture)
- `DOC/04_phase_routage_budget.md` — Phase 4 : Spectromètre (allocation)
- `DOC/05_phase_pareto.md` — Phase 5 : Validation (tests)

### **Méthodologie & état (2 fichiers)**
- `DOC/FALSIFIABILITE.md` — 13 règles go/no-go, hiérarchie risques
- `DOC/carnet_de_bord.md` — Journal vivant (hypothèses, décisions, avancement)

---

## 🏗️ Code & infrastructure

- **`CODE/`** — Modules Python par phase (shared, infra, catalog, phases 1-5)
- **`OPS/`** — Configuration, scripts, logs, checkpoints
- **`ARBORESCENCE.md`** — Structure complète CODE/DOC/OPS

---

## 🎯 Deux livrables

### **Partie 1 — Science fondamentale** ✅
Classification exhaustive + batterie de tests des propriétés mathématiques de l'attention.

Catalogue : [`DOC/CATALOGUE.md`](DOC/CATALOGUE.md) (131 propriétés × 18 catégories).

🟢 Ne peut pas "fail" — chaque mesure enrichit la classification. Publication indépendante possible.

### **Partie 2 — Validation hypothèse polymorphique** 🔄
Application de la Partie 1 pour vérifier si l'**attention sub-quadratique via allocation dynamique** est viable.

Phases 1–5 : testées séquentiellement, chacune a des critères go/no-go explicites.

---

## 🔗 Liens rapides

| Besoin | Fichier |
|--------|---------|
| **Débuter** | `DOC/INTRODUCTION.md` |
| **Comprendre la science** | `DOC/00_FONDATIONS.md` |
| **Voir le catalogue** | `DOC/CATALOGUE.md` |
| **État courant + next steps** | `ROADMAP.md` |
| **Hypothèses & décisions** | `DOC/carnet_de_bord.md` |
| **Critères succès/échec** | `DOC/FALSIFIABILITE.md` |
| **Structure code** | `ARBORESCENCE.md` |

---

## 📊 État du projet (2026-05-12)

| Phase | État | Blocker |
|-------|------|---------|
| **Catalogue Partie 1** | ✅ Prioritaire | En développement (Sprint A1) |
| **Phase 1** | ✅ Complet | 119 tests CPU passent |
| **Phase 1.5** | ✅ Complet | Signal S_Spectral validé |
| **Phase 2** | ✅ En cours | SCH corroborée (r_eff médian=2) |
| **Phase 3** | 🔄 Dev | Await pod RTX 5090 |
| **Phase 4** | 📋 Planifié | — |
| **Phase 5** | 📋 Planifié | — |

**Stack** : PyTorch + Fabric + Hydra + uv | **Hardware** : RunPod RTX 5090 (+ VPS dev)

---

## 📝 Citation

Projet fondateur en 2025. Deux livrables distincts : (1) classification mathématique des propriétés d'attention, (2) démonstration de l'allocation dynamique de rang en O(N) en moyenne.

---

**Documentation consolidée 2026-05-12** — 15 fichiers → 11 fichiers (-27% fragmentation)

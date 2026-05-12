# Introduction — Documentation ASP

**Bienvenue.** Ce document est votre **point d'entrée unique** dans la documentation du projet Attention Superlinéaire Polymorphe (ASP).

---

## 🎯 Cadrage en 2 livrables

Le projet produit **deux livrables scientifiques distincts** :

### **Partie 1 — Science fondamentale** 
Étude spectrale exhaustive des propriétés mathématiques de l'attention dense. Livrable : **classification + batterie de tests** réutilisable par toute la communauté.

Catalogue exhaustif : [`CATALOGUE.md`](CATALOGUE.md) (~131 propriétés × 23 catégories A-W + N).

**État implémentation** : **131 Properties codées sur 23 familles A-W + N** (catalogue complet, fin session 2026-05-12). 5 Oracle adapters (Synthetic, SMNIST, LL, Vision, Code), 10 Sprint runners, 6 scripts de génération livrables.

🟢 **Ne peut pas "fail"** — chaque mesure (positive ou négative) enrichit la classification. Publication indépendante possible : *"Mathematical Signatures of Attention Across Domains"*.

### **Partie 2 — Validation hypothèse polymorphique (ASP)**
Application de la Partie 1 pour vérifier si l'**attention sub-quadratique via allocation dynamique guidée par signal observable** est viable.

Conditionné à la réussite de la Partie 1. Alternative parmi d'autres (kernel approx, sparse, low-rank, state-space).

🟡 **Peut "fail"** — c'est acceptable scientifiquement. Chaque echec ferme une voie et oriente la suivante.

---

## 📚 Table des matières

### **Fondations** 
- [`00_FONDATIONS.md`](00_FONDATIONS.md) — Thèse ASP, vocabulaire architectural + mathématique

### **Catalogue & Oracles** (Partie 1)
- [`CATALOGUE.md`](CATALOGUE.md) — 131 propriétés, sélection 6 Oracles, batteries, prédictions a priori

### **Critères d'évaluation**
- [`FALSIFIABILITE.md`](FALSIFIABILITE.md) — 13 règles go/no-go par phase, hiérarchie risques

### **Sprints orchestration** (orchestre Partie 1 + Partie 2)
- [`sprints/README.md`](sprints/README.md) — 10 Sprints (B, C, D, E, F, G, S4-S7) avec compute estimé

### **Paper outlines**
- [`paper/README.md`](paper/README.md) — Plan Partie 1 + Partie 2 + bibliography + génération auto figures

### **Phases scientifiques** (Partie 2)
| # | Phase | Fichier |
|---|-------|---------|
| 1 | Métrologie (RCP/SSG) | [`01_phase_metrologie.md`](01_phase_metrologie.md) |
| 1.5 | Identification Gate (signaux) | [`01b_phase_calibration_signal.md`](01b_phase_calibration_signal.md) |
| 2 | Audit spectral (SCH) | [`02_phase_audit_spectral.md`](02_phase_audit_spectral.md) |
| 3 | ASPLayer (architecture) | [`03_phase_kernel_asp.md`](03_phase_kernel_asp.md) |
| 4 | Spectromètre (allocation) | [`04_phase_routage_budget.md`](04_phase_routage_budget.md) |
| 5 | Validation (tests) | [`05_phase_pareto.md`](05_phase_pareto.md) |

### **État & Processus**
- [`carnet_de_bord.md`](carnet_de_bord.md) — Journal vivant (hypothèses, décisions, avancement chronologique)

### **Architecture**
- [`adr/`](adr/) — Architecture Decision Records (décisions clés : stack, hardware, logging)

### **Rapports**
- [`reports/`](reports/) — Rapports générés par phase (phase2.md existant, autres en templates)

---

## 🚀 Par où commencer ?

### **Scientifique (comprendre le "quoi")**
1. Lire cette page (vous y êtes)
2. [`00_FONDATIONS.md`](00_FONDATIONS.md) — la thèse en 20 min
3. [`CATALOGUE.md`](CATALOGUE.md) § Propriétés — les 131 mesures (toutes implémentées)
4. [`FALSIFIABILITE.md`](FALSIFIABILITE.md) — critères succès/échec
5. [`paper/partie1/outline.md`](paper/partie1/outline.md) — plan publication Partie 1

### **Phases (comprendre le "comment")**
Commencer par phase 1 : [`01_phase_metrologie.md`](01_phase_metrologie.md)  
→ Lire 01, 01b, 02 en séquence (chacune dépend de la précédente)  
→ Phases 03-05 possibles seulement si 02 passe ses critères

### **Opérationnel (lancer le code)**
- [`../ROADMAP.md`](../ROADMAP.md) — État courant + prochaines actions
- [`../ARBORESCENCE.md`](../ARBORESCENCE.md) — Structure CODE/DOC/OPS
- [`sprints/README.md`](sprints/README.md) — Lancement Sprints pod (commandes prêtes)
- [`../OPS/setup/SETUP.md`](../OPS/setup/SETUP.md) — Bootstrap pod RunPod RTX 5090

---

## 📖 Légende des documents

| Type | Exemple | Rôle |
|------|---------|------|
| **Thèse** | 00_FONDATIONS | Explication scientifique, concepts |
| **Protocole** | 01-05_phase_* | Description d'une phase expérimentale |
| **Catalogue** | CATALOGUE | Mesures et propriétés ; oracle selection |
| **Journal** | carnet_de_bord | Trace du processus, hypothèses testées |
| **Méthodologie** | FALSIFIABILITE | Critères d'arrêt, règles de sévérité |
| **Implémentation** | ../ROADMAP, ../ARBORESCENCE | État courant, structure code |

---

## ❓ Questions fréquentes

**Pourquoi deux livrables ?**  
→ Si la Partie 2 (ASP) échoue, la Partie 1 (catalogue) reste un résultat scientifique solide et publiable. Découpler maximise la valeur attendue.

**Qu'est-ce qu'un "Oracle" ?**  
→ Un Transformer dense entraîné de zéro sur un domaine. Il fourni la **borne supérieure** que l'ASP tente d'atteindre. Cf. [`CATALOGUE.md`](CATALOGUE.md) § Oracles.

**Comment la phase 1.5 est-elle "gatekeep" ?**  
→ Si aucun signal parmi {S_KL, S_Grad, S_Spectral} n'exhibe corrélation > 0.70 avec le stress *et* insensibilité au bruit, le projet ASP s'arrête. Parties 1 + 2 ne peuvent pas continuer. Cf. [`FALSIFIABILITE.md`](FALSIFIABILITE.md) § Hiérarchie risques.

**Y a-t-il des redites dans la doc ?**  
→ Non par design. Chaque fichier a un rôle distinct. Vocabulaire : consultez [`00_FONDATIONS.md`](00_FONDATIONS.md).

---

## 🔗 Liens rapides

- **Vocabulaire architectural** (`Property`, `Battery`, `Oracle`, etc.) → [`00_FONDATIONS.md`](00_FONDATIONS.md) § Vocabulaire architectural
- **Vocabulaire mathématique** (`Hankel`, `r_eff`, `SCH`, etc.) → [`00_FONDATIONS.md`](00_FONDATIONS.md) § Vocabulaire mathématique
- **Code source** → [`../ARBORESCENCE.md`](../ARBORESCENCE.md)
- **État du projet + commandes** → [`../ROADMAP.md`](../ROADMAP.md)
- **Chronologie & hypothèses** → [`carnet_de_bord.md`](carnet_de_bord.md)

---

**Version** : 2026-05-12 fin session | **Consolidation** : 4 fichiers fusionnés en 1 point d'entrée | **Code** : 719 tests verts, **131 Properties** (catalogue complet), scaffolding complet ready-to-pod

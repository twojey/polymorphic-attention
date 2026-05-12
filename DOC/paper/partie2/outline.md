# Partie 2 — ASP : Polymorphic Attention via Signal-Guided Routing

> **Type** : article méthode + validation empirique.
> **Public** : NeurIPS / ICML (méthode), TMLR (négatif instructif si NO-GO).
> **Statut** : conditionné à Sprint G verdict (GO / PARTIAL / NO-GO).

## Abstract (draft)

We propose **Attention Superlinéaire Polymorphe (ASP)** : a meta-architecture where attention's algorithmic class is selected dynamically per-input rather than fixed at design time. ASP combines (i) a **Spectromètre** module that infers stress signals (ω/Δ/ℋ) from context, and (ii) an **ASPLayer** that routes the computation to one of K parameterized attention realizations (dense, butterfly, banded, low-rank, ...). We validate the approach on SMNIST + Oracle from Partie 1, comparing val_acc and inference cost against Oracle baseline and fixed-class baselines.

**Result** (à remplir après Sprint G) :
- **GO** : "We achieve <Y>% Oracle val_acc with <X>× FLOPs reduction at R_max = r_med/2..."
- **PARTIAL** : "Identification (5a) works, autonomous routing (5c) partial, OOD fragile (5e)..."
- **NO-GO** : "Despite identifying r_target from context, routing fails when R_max < r_med ; we discuss the failure modes (Sprint G report) and propose..."

## 1. Introduction

- Existing efficient attention picks ONE structure (sparse, low-rank, kernel) at design time.
- Hypothesis : attention's effective class **varies per-input**. A model that picks the right class dynamically should outperform any fixed choice.
- Pre-condition : Partie 1 must show that some structural diversity exists per régime.

## 2. Method

### 2.1 ASPLayer
- Input : keys K, queries Q, values V, signal r_pred
- Mix of K realizations weighted by routing distribution
- Differentiable end-to-end

### 2.2 Spectromètre
- Input : Q · Kᵀ (raw scores)
- Output : r_pred (predicted r_target)
- Trained with auxiliary L_R loss (predicts ground-truth r in warm-up)

### 2.3 Training schedule
- Phase 4a (Sprint E) : warm-up with ground-truth r
- Phase 4b (Sprint F) : autonomous (no ground-truth signal at inference)
- Phase 5 (Sprint G) : validation suite

### 2.4 R_max budget
- R_max = r_med / 2 (cible Sprint 6c) : "ASP doit faire mieux que la moyenne avec moitié des paramètres"

## 3. Experiments

### 3.1 SMNIST controlled setup (Sprint D)
- Oracle baseline 0.645 val_acc
- ASP V3 result : ...

### 3.2 Identification (test 5a)
- R_max ↔ r_target prediction quality

### 3.3 Self-emergent half-truth (test 5c)
- Spectromètre converge sans ground-truth signal

### 3.4 R_max sweep (test 6c)
- Quality vs budget Pareto

### 3.5 Anti-fraud + OOD (5d, 5e)
- ASP n'utilise pas de "shortcut" facile
- Tient sur données hors distribution

## 4. Analysis

### 4.1 What ASP actually routes to
- Visualisation des poids de routage par classe
- Corrélation avec ground-truth r

### 4.2 Failure modes (si applicable)
- Sprint G NO-GO : pourquoi ?
- Quelles classes échouent à être routées ?

### 4.3 Cost-quality Pareto
- ASP vs fixed-class baselines (dense, sparse, butterfly)
- ASP vs Oracle

## 5. Discussion

- Strengths and limits
- Comparison to MoE (Mixture of Experts) — ASP is "MoE of attention realizations"
- When ASP wins, when it doesn't

## 6. Conclusion

- ASP est viable / partiel / rejeté
- Implications pour la design d'efficient attention

## Bibliography

(Voir DOC/paper/README.md §Bibliography.)

## Figures

- Figure 1 : ASP architecture diagram
- Figure 2 : training curves Sprint E + F
- Figure 3 : R_max sweep quality
- Figure 4 : Pareto curves ASP vs baselines
- Figure 5 : routing weights visualisation
- Figure 6 : OOD robustness

## Tableaux

- Tableau 1 : Configuration ASP (hyperparams)
- Tableau 2 : Comparaison Oracle / ASP / baselines
- Tableau 3 : Résultats par test (5a-6c)
- Tableau 4 : Compute / latency comparison

## Annexes

- A. ASPLayer implementation details
- B. Spectromètre architecture
- C. Reproducibility (sprint configs, checkpoints, seeds)

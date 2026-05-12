# Templates Sprints S4-S7 — Oracles cross-domain

> Sprints d'extension de la Partie 1 vers d'autres domaines : Vision, Code, LL,
> SMNIST long-range. Format mutualisé.

## Sprint S4 — SMNIST seq_len étendu

- **Configs delta** : 1024 / 2048 / 4096
- **Cible** : valider que les signatures (r_eff, B5 block-diag, etc.) restent stables ou évoluent prédictiblement avec delta
- **Critère** : r_eff(delta) bien décrit par loi de transfert phase 2 V1

## Sprint S5 — Vision (DINOv2)

- **Backend** : HuggingFace `facebook/dinov2-base` (n_layers=12, n_heads=12)
- **Régimes** : patch_size × n_classes × image_complexity
- **Critère** : H-matrix structure attendue (Q2 ≤ 5, B5 block-diag fort)
- **Prédiction confrontée** : DINOv2 est le candidat le plus low-rank (DOC/CATALOGUE §4.2)

## Sprint S6 — Code (StarCoder / MinimalCode + Dyck-k)

- **Backend** : `bigcode/starcoder2-3b` OU MinimalLM trained
- **Régimes** : depth × seq_len × n_bracket_types
- **Critère** : rang Hankel fini (P1 ≤ depth + 1), sparsité forte (B4 > 0.5)
- **Prédiction confrontée** : StarCoder = haut r_eff (DOC/CATALOGUE §4.2)

## Sprint S7 — LL (Llama-3.2-1B / TinyStories)

- **Backend** : `meta-llama/Llama-3.2-1B` OU MinimalLM trained
- **Régimes** : depth (parenthèses imbriquées) × seq_len
- **Critère** : nestedness Q4 préservée vs depth = clé pour Sprint D phase 3
- **Prédiction confrontée** : LL low-rank effectif, pas Mercer (DOC/CATALOGUE §4.2)

## Format de rapport commun

Chacun des Sprints S4-S7 produit :
- `summary.json` : metrics + go/no-go decisions
- `results.json` : Battery × régimes Oracle
- `report.md` : table Property × régime
- `predictions_evaluation.md` : confrontation paris a priori vs mesures

Pour le rapport cross-Oracle global, voir `DOC/paper/partie1/`.

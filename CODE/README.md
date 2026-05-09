# CODE — Implémentation par phase

Squelette par phase. Aucune implémentation au démarrage : la stack (PyTorch / JAX / autre) est arrêtée à la fin de la phase 1 et fixée ici par un fichier de conventions.

## Arborescence

```
CODE/
├── phase1_metrologie/             # RCP via SSG : Oracle, rang Hankel, entropie spectrale sur (ω, Δ, ℋ)
├── phase1b_calibration_signal/    # Identification Gate : 4 signaux candidats, sensibilité + immunité, distillabilité
├── phase2_audit_spectral/         # Audit Spectral : SVD → r_eff, Stress-Rank Map, loi de transfert (SCH)
├── phase3_kernel_asp/             # ASPLayer : Backbone + correction Matriochka + Loss Consistency, Soft-Mask
├── phase4_routage_budget/         # Spectromètre : 4a (warm-up + distillation) → 4b (autonome), Diagramme de Phase
└── phase5_pareto/                 # Validation : 5 tests (Identifiabilité, Élasticité, SE/HT, Pareto, OOD croisé)
```

Chaque sous-dossier contient un README qui décrit l'attendu de la phase et les entrées/sorties prévues.

## Conventions (à compléter post-phase 1)

- Stack : *à fixer*
- Versionnage des poids : *à fixer*
- Format des matrices d'attention extraites : *à fixer*
- Logging des expériences : *à fixer* (cf. `OPS/`)
- Reproductibilité (seeds, hardware fingerprint) : *à fixer*

## Règle d'ordre

Pas de code dans `phaseK/` tant que la phase K-1 n'a pas passé son go-criterion documenté dans `DOC/falsifiabilite.md`.

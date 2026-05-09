# OPS — Opérations, configurations, environnement

Tout ce qui n'est ni documentation ni code de modèle : configurations d'expérience, scripts d'orchestration, gestion de l'environnement, journalisation.

## Sous-arbre prévu (à créer au fil de l'eau)

```
OPS/
├── env/        # spécification d'environnement (lock files, container, hardware fingerprint)
├── configs/    # configurations d'expérience par phase (yaml/toml/json)
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   └── phase5/
├── scripts/    # scripts d'orchestration (run, sweep, export, packaging des résultats)
└── logs/       # logs et artefacts d'expérience (gitignored ou stockage externe)
```

## Conventions

- **Une config = une expérience.** Une config doit contenir tout ce qui est nécessaire pour rejouer un run sans toucher au code : seed, hyperparamètres, dataset, hardware ciblé, version de stack.
- **Pas de configs implicites.** Les défauts du code ne tiennent pas lieu de spécification. Toute valeur qui change un résultat doit apparaître dans une config.
- **Pré-enregistrement.** Les configs de toutes les phases sont versionnées *avant* le lancement (cf. règle 4 de DOC/falsifiabilite.md). Une config modifiée après run doit être copiée et renommée, jamais écrasée.
- **Reproductibilité.** Chaque run produit un manifest (seed, commit, fingerprint hardware, durée, hash des données). Sans manifest, le résultat ne compte pas.

## À faire post-phase 1

Choisir et documenter ici :
- Outil de gestion d'environnement (uv, poetry, conda, container OCI).
- Outil de logging d'expérience (wandb, mlflow, tensorboard, fichiers locaux).
- Conventions de nommage des runs et artefacts.

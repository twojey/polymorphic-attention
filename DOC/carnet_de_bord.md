# Carnet de bord — projet ASP

Journal vivant du projet *Attention Superlinéaire Polymorphe*. Capture le processus : décisions au fil de l'eau, hypothèses à tester, surprises, bugs résolus, avancement.

> **Distinction avec les rapports phase** (`DOC/reports/phase{1,1b,2,3,4,5}_template.md`) : ces rapports sont les conclusions finales d'une phase. Ce carnet est **chronologique et bordélique**, il sert à se souvenir de comment on est arrivé à ces conclusions, et à tracer les hypothèses qu'on n'a pas encore testées.

---

## Comment utiliser ce carnet

- **Append-only chronologique** dans la section *Avancement* — nouvelles entrées en haut.
- **Hypothèses, décisions, surprises** : sections thématiques, on édite au fur et à mesure.
- **Quand une hypothèse devient résultat consolidé** : déplacer vers le rapport de phase correspondant, garder un pointeur ici.
- **Tag des entrées** : utiliser `#hypothese`, `#decision`, `#bug`, `#surprise`, `#milestone` pour rechercher facilement.
- **Liens MLflow** : citer les `run_id` (8 premiers chars du run_uuid) pour traçabilité.

---

## Hypothèses à tester (ordre de priorité)

| # | Hypothèse | Phase test | Statut |
|---|---|---|---|
| H1 | Les matrices d'attention de l'Oracle dense exhibent un rang Hankel ≪ N **ou** une entropie spectrale ≪ log N sur une portion non-triviale du sweep SSG | 1 | en cours (run e2f0b5e) |
| H2 | Au moins un signal local parmi {S_KL, S_Grad, S_Spectral} prédit le rang structurel avec ρ_Spearman > 0.70 sur ω ou Δ, ET ρ_ℋ < 0.20 | 1.5 | à tester post-phase 1 |
| H3 | La SCH se vérifie comme distribution avec IQR raisonnable par rapport à la médiane (V3.5 : pas comme fonction) | 2 | à tester post-phase 1.5 |
| H4 | Le catalogue {Toeplitz, Hankel, Cauchy, compositions} couvre la majorité des régimes (ε_C résiduel < 0.30) | 2 | à tester post-phase 1.5 |
| H5 | La loi de transfert `r_eff = a × (1+ω)^α × (1+Δ)^β × exp(γ·ℋ)` a des exposants reproductibles | 2 | à tester post-phase 1.5 |
| H6 | Les exposants sont **universels cross-domain** (verdict `cross_domain_compare`) | 2 | Sprint 4 (multi-Oracle) |
| H7 | Test 6c : ASP avec R_max = r_med/2 atteint ≥ 95 % qualité Oracle | 5 | Sprint 4 |

## Hypothèses fortes / "résultats Tier S" attendus si protocole va au bout

- **Loi universelle** : exposants α, β identiques cross-domain → physique de l'attention
- **Classe hors catalogue** (batterie D) : famille structurée non répertoriée
- **R_max/2 strict** : preuve quantifiée que l'attention dense est sur-paramétrée d'un facteur 2

Cf. discussion exhaustive 2026-05-10 (avancement).

---

## Décisions actées (chronologique inverse)

### 2026-05-10 — `extraction.batch_size=4` au lieu de 16
**#decision** Avec `max_seq_len=8192` et capture_attn=True, peak mémoire = 6 layers × 16 batch × 8 heads × N² × 2 bytes = **38 GB → OOM** sur 32 GB. Override CLI à 4 jusqu'à refactor extract.py per-layer.
**Conséquence** : seulement 4 matrices A extraites par run V1. Statistiques par régime limitées. À adresser dans driver V2.

### 2026-05-10 — `max_seq_len 4096 → 8192` dans config phase 1
**#decision** Le commentaire « ≥ Δ_max prévu en sweep » du config sous-estimait la formule SSG : `seq_len ≈ 1+1+ω×(2Δ+2)+1`. Pour ω=2, Δ=1024 : ~5100. Marge x1.6 à 8192. Commit `fc834a6`.

### 2026-05-10 — Tunnel SSH inverse VPS→pod (vs forward depuis pod)
**#decision** Le ROADMAP suggérait un forward tunnel pod→VPS, mais ça nécessite une clé SSH du pod vers le VPS. Le reverse tunnel utilise la clé existante VPS→pod, plus simple. Script `start_mlflow_tunnel.sh`.

### 2026-05-10 — MLflow `--serve-artifacts` (proxy mode)
**#decision** Sans ce flag, le client MLflow tente d'écrire les artefacts directement au filesystem du serveur — qu'il n'a pas (pod ≠ VPS). Avec `--serve-artifacts` + `--default-artifact-root mlflow-artifacts:/`, le client upload via HTTP. Découvert au smoke test phase 1.

### 2026-05-10 — Copie clé SSH dans `OPS/secrets/` + gitignored
**#decision** Pour qu'un nouveau VPS ou poste de travail puisse atteindre les pods sans recréer de clé. Couvert par règle `secrets/` du `.gitignore` racine.

### 2026-05-10 — Splitter `setup_env.sh` en scripts modulaires dédiés
**#decision** Un script = une responsabilité. `blackwell_env.sh` (ENV vars), `install_uv.sh`, `install_python_deps.sh`, `verify_torch.sh`, `setup_pod.sh` (orchestrateur). Documenté dans `OPS/env/SETUP.md`. setup_env.sh devient wrapper de compat.

### 2026-05-10 — Sprint 1 mono-Oracle (SMNIST seul)
**#decision** ROADMAP §stratification : multi-Oracle (code synthétique, vision) reporté Sprint 4. Permet d'avoir un livrable autonome après chaque sprint.

### 2026-05-10 — `extraction.batch_size=4` figé tant que extract.py n'est pas refactor per-layer
Idem ci-dessus, redondant — à fusionner après le run.

---

## Surprises et pièges (chronologique inverse)

### 2026-05-10 — Buffering stdout en file redirect
**#surprise** Avec `nohup ... > log 2>&1 &`, les `print()` Python sont block-buffered → log vide pendant des minutes. **Fix** : `python -u` ou `PYTHONUNBUFFERED=1`. Ajouté au launch command. À retenir pour tous les futurs nohup.

### 2026-05-10 — Le mécanisme checkpoint Oracle existait mais n'était pas wiré
**#bug** `train_oracle()` accepte `checkpoint_path: Path | None = None` et appelle `fabric.save()` à chaque amélioration de val_loss. Mais `run.py` n'a jamais passé de `checkpoint_path`. Patché `e2f0b5e` : `Path("/tmp")/f"{run_id}_oracle.ckpt"` + `mlflow.log_artifact()`.
**Leçon** : toujours grep le projet avant de réimplémenter.

### 2026-05-10 — Hydra defaults : groupe vs include
**#bug** `defaults: - thresholds_phase1: thresholds_phase1` cherche `thresholds_phase1/thresholds_phase1.yaml` (sous-dir). Les fichiers étaient siblings de `oracle_smnist.yaml` → ne loadent pas. **Fix** : syntaxe `@namespace` :
```yaml
defaults:
  - _self_
  - thresholds_phase1@thresholds_phase1
  - dataset_split_smnist@dataset_split_smnist
```

### 2026-05-10 — `torch.__version__` n'est pas une str
**#bug** PyTorch 2.x retourne un objet `TorchVersion` qui print comme `'2.11.0+cu128'` mais n'est pas sérialisable par `yaml.safe_dump`. Idem `torch.version.cuda`. **Fix** : cast `str()` dans `hardware_fingerprint()`.

### 2026-05-10 — `collate()` assumait seq_len uniforme
**#bug** Le commentaire : « Ici on assume que la position QUERY est la même pour tous les exemples du batch (vrai dans un sweep monovarié à ω/Δ fixés). ». Mais le ConcatDataset mélange tous les sweeps → seq_len variable → `torch.stack` échoue. **Fix** : `pad_sequence` avec `pad_id` du Vocab.

### 2026-05-10 — Extraction extracted matrices pas alignées sur device
**#bug** Le modèle wrapped Fabric est sur cuda, mais le DataLoader d'extraction reste CPU. Forward planté. **Fix** : `tokens.to(device)` + `query_pos.to(device)` dans `AttentionExtractor.extract()`.

---

## Avancement chronologique

### 2026-05-10 lundi (après-midi) — Sprint 1 démarrage

#### 14:35 Paris — Phase 1 run en cours, overfit observé
- Run `s1_smnist_oracle_e2f0b5e` à epoch 9.
- Best val_loss = 1.49 (epoch 4), checkpoint sauvegardé.
- Depuis epoch 4 : val_loss diverge (1.68 → 1.81 → 2.06 → 2.10) malgré train_loss continue ↓ (0.39).
- val_acc 62.6 % stabilise.
- **Plateau attendu vers epoch 12 (~15:05 Paris).**
- **Discussion** : overfit attendu sur SMNIST simple. Oracle = borne sup, pas optimum de généralisation. Le checkpoint d'epoch 4 sera utilisé pour phase 1.5. Pas un problème scientifique.

#### 13:35 Paris — script monitor_run.sh créé
- Encapsule la logique de check (MLflow + pod SSH + GPU) dans un script unique avec codes de sortie 0/1/2/3.
- Cron `06416aae` mis à jour : `bash OPS/scripts/monitor_run.sh` toutes les 15 min (:07/:22/:37/:52).
- Commit `5222121`.

#### 12:53 Paris — relance run phase 1 avec checkpoint patch
- Premier run `s1_smnist_oracle_fc834a6` killed à 25 min (epoch 0 done).
- Patch wire `checkpoint_path` à `train_oracle()` + `mlflow.log_artifact()` après training.
- Smoke test : checkpoint 19 MB uploadé via tunnel → VPS disque.
- Commit `e2f0b5e`. Restart avec `PYTHONUNBUFFERED=1` → epochs visibles en live.

#### 11:00-12:30 Paris — restructuration scripts setup
- Création `blackwell_env.sh` (source-only), `install_uv.sh`, `install_python_deps.sh`, `verify_torch.sh`, `setup_pod.sh`, `start_mlflow_tunnel.sh`.
- `OPS/env/SETUP.md` : walkthrough complet pod-from-scratch + troubleshooting.
- Clé SSH copiée dans `OPS/secrets/` (gitignored).
- Mémoire `reference_setup_docs.md` indexée dans `MEMORY.md`.
- Commit `38fc01b`.

#### 10:30 Paris — premier smoke test phase 1 + bugs
- Bugs découverts : Hydra defaults, torch.__version__, collate, extract device, max_seq_len.
- Tous fixés en série, smoke test passe (3 epochs en ~13s).
- Optimisations GPU ajoutées : `torch.compile(dynamic=True)`, DataLoader `num_workers=4 + pin_memory`, `set_float32_matmul_precision("high")`. Gain ~30 % sur hot path.
- Commits `651d5e7`, `fc834a6`.

#### 10:00 Paris — setup pod terminé
- Pod LUMIS-SFT-3 (RTX 5090, sm_120, 32 GB, driver 570.153) accessible.
- `setup_env.sh` (alors monolithique) lance avec succès. PyTorch 2.11.0+cu128 installé.
- 6/6 primitives Blackwell passent (`validate_primitives.py`).
- Tunnel MLflow inverse VPS→pod opérationnel (`--serve-artifacts` mode proxy).

---

## Questions ouvertes

- **Q1** : Quand on aura les premières matrices A extraites, la SCH se manifestera-t-elle déjà avec seulement 4 exemples ? Probablement non statistiquement, mais qualitativement on verra si rang Hankel ≪ N émerge.
- **Q2** : V1 driver fait `break` après 1 batch d'extraction → seulement 4 matrices. Pour Sprint 1 verdict, est-ce suffisant ? Ou faut-il modifier le driver pour boucler sur audit_svd entier ?
- **Q3** : Si phase 1.5 GO, faudra-t-il refaire phase 1 avec extraction étendue (loop sur tout audit_svd) avant phase 2 ? Probablement oui.
- **Q4** : Le `extraction.batch_size=4` actuel limite la batterie de stat. Refactor extract.py per-layer avant Sprint 2 (phase 2 = audit spectral).

---

## Liens utiles

- Spec V3.5 : `DOC/00_vision.md` à `DOC/05_phase_pareto.md`
- Glossaire : `DOC/glossaire.md`
- Falsifiabilité : `DOC/falsifiabilite.md`
- Templates rapports : `DOC/reports/phase{1,1b,2,3,4,5}_template.md`
- Setup pod : `OPS/env/SETUP.md`
- ROADMAP exécution : `ROADMAP.md`
- MLflow : http://localhost:5000 (via tunnel SSH depuis poste local)

# CONTEXT — Vocabulaire architectural du projet ASP

Définitions opérationnelles pour les **modules architecturaux** du codebase. Vocabulaire utilisé dans le code, les rapports et les commits. Distinct de `DOC/glossaire.md` qui couvre le **vocabulaire mathématique** (matrice de Toeplitz, r_eff, Ho-Kalman, etc.).

Termes liés à l'architecture du code : [LANGUAGE.md](/.claude/skills/improve-codebase-architecture/LANGUAGE.md) (Module, Interface, Seam, Adapter, Depth, Leverage, Locality).

---

## Concepts du catalogue (`CODE/catalog/`)

### Property

**Définition** — un module qui calcule UNE mesure mathématique sur une (ou plusieurs) matrice(s) d'attention.

Une Property est :
- **Identifiée** : `name` (ex `A1_r_eff_theta099`), `family` (ex `A`)
- **Datée en coût** : `cost_class` ∈ {1, 2, 3, 4, 5} (1 = rapide < 1s/régime, 5 = lent > 1min)
- **Typée en pré-requis** : `requires_fp64`, `requires_symmetric`, `scope` (per_regime | cross_regime)
- **Single-responsibility** : retourne `dict[str, float | str]`, NE logge PAS à MLflow elle-même

Une Property = ~30-80 lignes de Python, un fichier dédié.

Le catalogue DOC/00b spécifie 131 Properties (A1-W6). Le code projet vise à en implémenter ~60 (priorité haute + médium) en Sprint A.

### Family

**Définition** — regroupement sémantique de Properties qui partagent un cadre théorique (rang de déplacement, Ho-Kalman, équivariances, etc.).

Les Families sont les sections de DOC/00b. Couvertes par les sous-packages `catalog/properties/family_*/`.

### Projector

**Définition** — primitive mathématique réutilisable qui implémente UNE structure (Toeplitz, Hankel, Cauchy, Vandermonde, Butterfly, Monarch, etc.).

Un Projector expose deux opérations :
- `project(A: Tensor) → Tensor` : projection orthogonale (Frobenius) de A sur l'espace de la classe (utilisé par Properties pour calculer ε_C)
- `operator(params) → callable` : opérateur paramétré apprenable (utilisé par Backbone phase 3 le cas échéant)

Un Projector vit dans `catalog/projectors/<nom>.py` et est consommé par les Properties de la family correspondante.

### PropertyContext

**Définition** — contexte d'exécution partagé entre Properties d'un même run, jouant le rôle de **cache lazy** pour les pre-computations coûteuses.

Quand `B1_toeplitz_distance` et `B5_block_diagonal` veulent toutes deux la même SVD batchée, la première remplit le cache via `ctx.get_or_compute("svd", ...)` et la seconde réutilise. Évite la N²+ duplication entre Properties.

Le PropertyContext porte aussi :
- `device`, `dtype` : politique machine (résolue par `infra/machine.py`)
- `regime` : dict des paramètres de stress (ω, Δ, ℋ) en cours pour cette computation per_regime
- `attentions` : référence à la collection d'attentions (per_regime ou cross_regime selon scope)

### Battery

**Définition** — composition de Properties selon un **niveau** (minimal | principal | extended | full | research). Orchestre l'exécution sur un Oracle, l'agrégation distributionnelle (V3.5 RegimeStats), et le logging MLflow.

Niveaux :
- `level_minimal` : ~5 Properties, smoke check (~1 min wall-clock)
- `level_principal` : ~30 Properties, priorité haute (~10 min)
- `level_extended` : ~80 Properties, priorité med + haute (~1h)
- `level_full` : 131 Properties DOC/00b (~plusieurs heures)
- `level_research` : familles V (analytique frontière) + W (model theory)

Une Battery = un fichier `catalog/batteries/level_<nom>.py` qui énumère ses Properties.

### Oracle (Adapter)

**Définition** — Adapter qui fournit des matrices d'attention dense à la Battery, pour un **domaine** donné (SMNIST, LL, vision, code…).

Un Oracle expose :
- `extract_regime(regime: RegimeSpec, n_examples: int) → AttentionDump` : récupère un dump pour un point du sweep de stress
- `oracle_id`, `domain`, `n_layers`, `n_heads`, `vocab_size` : métadonnées

Le seam Oracle permet la **comparaison cross-domain** (SMNIST × LL × ...) qui est le cœur de la Partie 1 (livrable scientifique "Mathematical Signatures of Attention"). Implémentations : `SMNISTOracle` (wrappe phase 1 existant), `LanguageOracle` (à coder).

### AttentionDump

**Définition** — structure de données fixée du contrat Oracle ↔ Battery.

```
AttentionDump :
  attn:       list[L tensors (B, H, N, N) FP64]   # par couche
  omegas:     tensor (B,) int                      # stress récursion
  deltas:     tensor (B,) int                      # stress binding distant
  entropies:  tensor (B,) float                    # stress ℋ
  tokens:     tensor (B, N) long                   # input tokens (pour replay)
  query_pos:  tensor (B,) long                     # position du QUERY token
  metadata:   dict                                 # oracle_id, seed, etc.
```

---

## Concepts d'infrastructure (`CODE/infra/`)

### MachineProfile

**Définition** — Module qui répond à "comment optimiser sur cette machine ?" en un point d'autorité unique.

Auto-détecte :
- `device` : cuda / cpu
- `gpu_arch` : Blackwell sm_120 / Hopper / Ampere / ... (impacte choix FP64 vs FP32)
- `precision` : fp64 / fp32 (FP64 = 1/64 du FP32 sur consumer Blackwell, donc CPU FP64 ou GPU FP32)
- `batch_cap` : selon VRAM / RAM
- `n_threads_blas` : 1 forcé si fork multiprocessing actif, sinon all-cores

Consommé par tout code qui faisait `if cuda.is_available()` ou choisissait device/precision localement. Source unique de vérité.

---

## Concepts de scaffold (`CODE/shared/`)

### Driver Harness (à venir, candidat #6)

**Définition** — context manager qui factorise le boilerplate des drivers de phase : Hydra config → manifest → MLflow run → resource snapshot → cleanup → finalize.

Le Driver Harness éliminera ~150 lignes de scaffold par driver (4 drivers actuels = ~600 lignes économisées).

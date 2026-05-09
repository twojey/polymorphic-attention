# 04 — Phase 4 : Calibration du Spectromètre

L'ASPLayer de la phase 3 est *capable* d'allouer un rang variable. Cette phase lui donne une **intelligence de gestion** : on entraîne le Spectromètre à reconnaître la structure et à allouer le rang en conséquence, sous une contrainte budgétaire stricte.

Sans cette étape, le noyau ASP est une voiture puissante sans conducteur — il ouvrira tout, partout.

**Structure en deux temps (V2).** La phase 4 est explicitement scindée :
- **4a — Warm-up avec distillation** : la loss inclut un terme qui aligne α_t · R_max sur r_target = f(ω, Δ, ℋ) issu de la phase 2. Le Spectromètre apprend la *forme* attendue de l'allocation.
- **4b — Apprentissage autonome** : retrait *total* du terme de distillation. Le Spectromètre s'auto-calibre uniquement sur L_task + λ_budget · L_sparsity.

Ce découplage explicite isole le risque de fuite de labels (cf. section 8) et garantit qu'à l'évaluation phase 5, le Spectromètre ne dépend plus d'observables qu'il n'aurait pas en déploiement.

## 1. Le Spectromètre

Petit module (MLP léger ou conv 1D) qui scanne le flux latent x_t et prédit la valeur α_t ∈ [0, 1] du rang à allouer (projetée vers R_target en phase 3).

- **Entrée** — vecteur latent courant x_t et, optionnellement, un état récurrent court (les K dernières positions, ou un résumé glissant).
- **Sortie** — α_t scalaire, ou directement le vecteur de gates m_t ∈ [0, 1]^{R_max} monotone décroissant.

**Contrainte de capacité.** Le Spectromètre est volontairement petit (de l'ordre de quelques dizaines de milliers de paramètres) — voir aussi phase 5, test d'Identifiabilité. Trois raisons :

- s'il est gros, son coût annule l'économie qu'il est censé permettre,
- s'il est gros, il peut apprendre la *structure de la tâche* au lieu d'apprendre à *mesurer la surprise* — c'est exactement le risque de fraude que phase 5 cherche à débusquer,
- s'il est petit, son comportement reste interprétable.

## 2. Loss multi-objectif

Le Spectromètre n'est pas entraîné sur la seule perte de tâche. On lui impose une contrainte budgétaire :

```
L_total = L_task + λ_budget · L_sparsity
```

- **L_task** — perte classique (cross-entropy, MSE, …).
- **L_sparsity** — pénalité sur l'usage cumulé du rang. Forme canonique :
  ```
  L_sparsity = (1/T) Σ_{t=1..T} α_t        (équivalent : moyenne du rang utilisé)
  ```
  Variantes : pondération Matriochka (Σ_t Σ_k k · m_{t,k}, pénalise davantage les rangs élevés), L1 sur les gates, quadratique sur α_t.

- **λ_budget** — levier de contrôle.
  - λ_budget grand → modèle « paresseux » → efficience max, qualité à risque.
  - λ_budget faible → modèle « précis » → qualité max, gourmand.

**Recommandation initiale** : pondération Matriochka, cohérente avec la hiérarchie de phase 3 : ouvrir un rang élevé doit *coûter plus* qu'ouvrir un rang faible.

## 3. Continuité du gradient — STE et Gumbel-Softmax

Le choix d'un rang entier est non-dérivable. Deux mécanismes en plus du Soft-Masking de phase 3 :

### 3.1 Straight-Through Estimator (forme canonique)

Forward avec rang entier, backward avec gradient continu :

```
r_continuous = α_t * R_max
r_discrete   = round(r_continuous)
r_final      = r_continuous + (r_discrete - r_continuous).detach()
```

Le `.detach()` empêche le gradient de remonter par le terme arrondi ; le gradient passe entièrement par r_continuous. Forward propre, backward dérivable.

### 3.2 Gumbel-Softmax

Échantillonnage différentiable sur la distribution catégorielle des rangs, paramétré par une température τ. Avantage : explore vraiment l'espace des rangs pendant l'entraînement, pas juste le mode. Coût : optimisation plus délicate, sensibilité au planning de τ. À tenter en deuxième intention si le STE ne donne pas une politique stable.

## 4. Curriculum de Stress

On n'entraîne pas le Spectromètre sur l'ensemble du SSG simultanément. On utilise un curriculum à trois étages qui exploite explicitement la Stress-Rank Map et la loi de transfert produites en phase 2 :

### Étage 1 — Warm-up sur bruit structurel

Données : séquences à ℋ élevée (bruit blanc, distracteurs maximum). λ_budget *fort* dès le départ. Objectif : forcer le Spectromètre à apprendre que **par défaut, le rang est minimal**. C'est la base du test d'Identifiabilité (phase 5).

### Étage 2 — Injection structurelle légère

Introduction progressive de séquences à faible ω (récursion peu profonde) et faible Δ. Le modèle découvre que pour faire baisser L_task, il *doit* ouvrir un peu de rang. λ_budget maintenu, mais le Spectromètre apprend à associer la *présence d'une structure simple* à une montée modérée de α_t.

### Étage 3 — Stress maximal

Introduction des dépendances longues (Δ grand) et des récursions profondes (ω grand). Le Spectromètre doit maintenant apprendre l'**élasticité** : monter rapidement le rang sur les portions structurées, le redescendre dès que la structure se relâche.

**Pourquoi ce curriculum.** Si on commence directement à l'étage 3 avec λ_budget fort, le risque dominant est le **gel** : le Spectromètre s'éteint avant d'avoir appris le signal de surprise. Le curriculum garantit que le signal apparaît avant que la pression budgétaire ne le supprime.

### 4a — Warm-up avec distillation

Pendant les trois étages du Curriculum, on **active** un terme de distillation qui ancre α_t sur une cible issue de la phase 2.

**V3.5 — La cible est un percentile, pas une moyenne.** La SCH étant une distribution `P(r_eff | régime)` (pas une fonction), distiller sur la *moyenne* fait sous-allouer systématiquement sur les exemples atypiques (queue droite). On distille donc sur un percentile haut, par défaut **p75** :

```
r_target_p75(régime) = percentile_75( P(r_eff | régime) )       (depuis phase 2)
```

**V3.5 — Loss asymétrique.** Sous-allouer (α·R_max < r_target) coûte de la qualité ; sur-allouer (α·R_max > r_target) coûte du compute. L'asymétrie n'est pas symétrique. La distillation devient :

```
L_distil = λ_distil · [
    1[α·R_max < r_target_p75] · (α·R_max − r_target_p75)²   ← pénalise la sous-allocation fortement
  + γ · 1[α·R_max > r_target_p75] · (α·R_max − r_target_p75)²   ← pénalise la sur-allocation faiblement (γ < 1)
]
```

avec γ ∈ [0.1, 0.3] pré-enregistré. Asymétrie justifiée par : sous-allouer dégrade L_task ; sur-allouer ne dégrade que `L_sparsity` qui est déjà séparément contrainte par λ_budget.

**Loss totale phase 4a** :

```
L_total_4a = L_task + λ_budget · L_sparsity + L_distil(p75, asymétrique)
```

Sans cet ancrage, le Spectromètre risque de s'installer dans un minimum local trivial (par exemple, prédire une valeur constante de α qui minimise globalement L_task + λ_budget · L_sparsity sans corrélation avec la difficulté locale).

**Attention — fuite de labels.** (ω, Δ, ℋ) sont des paramètres de génération du SSG ; ce sont des labels que le modèle n'aura PAS à l'inférence. La distillation 4a est donc explicitement de la **supervision faible synthétique**, valide uniquement sur des tâches dont les paramètres de stress sont connus. Sur des données réelles (LM, etc.), 4a est sautée.

**Attention — l'Oracle n'est pas le maître.** r_target_p75 est ce que **l'Oracle utilise**, pas ce qui est *optimal*. L'Oracle peut sur-utiliser sa capacité par défaut de softmax non-régularisée. La distillation 4a sert d'amorce, pas de plafond. La phase 4b et le test "R_max réduit" de phase 5 (DOC/05 section 6c) doivent permettre à l'ASP de descendre *sous* r_target si possible.

### 4b — Apprentissage autonome

**Retrait total** du terme de distillation : λ_distil → 0. La loss devient :

```
L_total_4b = L_task + λ_budget · L_sparsity
```

Le Spectromètre doit maintenant prédire α_t à partir de **x_t seul** (et du signal de surprise validé en phase 1.5), sans aucune information sur (ω, Δ, ℋ). C'est dans cette configuration qu'il sera évalué en phase 5.

### Critère de transition 4a → 4b (pré-enregistré)

Trois conditions, toutes nécessaires :

1. **Convergence de L_distillation** — le terme ‖α·R_max − r_target‖² atteint un plateau (variation < seuil sur N époques).
2. **Diagramme de Phase initial cohérent** — la corrélation Spearman entre R_target prédit par le Spectromètre et r_target théorique dépasse 0,80 sur le set de validation.
3. **Pas de plafond artificiel** — l'accuracy n'est pas plafonnée par la distillation (vérifier qu'à λ_distil = 0 ponctuellement, la qualité ne dégrade pas).

Si ces trois conditions sont réunies, on bascule en 4b. Sinon, on reste en 4a et on diagnostique.

### Sanity check de basculement

À l'instant de bascule 4a → 4b, on mesure la dérive : variation de R_target moyen sur les N premiers steps de 4b. Une dérive importante (> 30%) signale une dépendance trop forte du Spectromètre à la distillation — risque de chute de qualité aval.

## 5. Le Diagramme de Phase

Livrable visuel principal de cette phase. À la fin de l'entraînement, on trace :

- **Axe X** — complexité réelle de la tâche (stress imposé par le SSG, ou r_target prédit par la loi de transfert phase 2).
- **Axe Y** — R_target choisi par le Spectromètre.

**Succès de la phase 4** : la courbe est une fonction croissante, idéalement proche de la diagonale (à un facteur d'échelle près). Le Spectromètre *anticipe* la structure : par exemple, dès qu'une parenthèse ouvrante apparaît dans une tâche de Dyck, R_target doit monter avant la parenthèse fermante correspondante.

Variantes du diagramme :
- Diagramme de Phase λ_budget — comment la courbe change quand on bouge λ_budget. Identifie les plages où le polymorphisme est effectif.
- Diagramme temporel — R_target en fonction de la position t pour des séquences typées (utile pour préparer le test d'Élasticité de phase 5).

## 6. Protocole

1. Pré-entraîner l'ASPLayer (phase 3) avec Spectromètre figé à α_t = 1 ∀t, pour caler le Backbone et les bases Matriochka sur la tâche.
2. **Phase 4a (warm-up avec distillation)** — activer le Spectromètre, λ_distil > 0. Lancer le curriculum (étages 1 → 2 → 3) avec λ_budget choisi.
3. Vérifier les trois conditions de transition 4a → 4b. Si non remplies, diagnostiquer.
4. **Phase 4b (apprentissage autonome)** — λ_distil = 0. Continuer l'entraînement jusqu'à convergence sur L_task + λ_budget · L_sparsity uniquement.
5. Répéter (4a + 4b) pour 5–7 valeurs de λ_budget (échelle logarithmique, pré-enregistrées).
6. Pour chaque λ_budget, mesurer : L_task finale, R_target moyen, distribution de R_target, corrélation R_target ↔ stress local construit.
7. Tracer le Diagramme de Phase **et** la courbe Pareto (qualité, complexité). Reporter la dérive 4a → 4b par valeur de λ_budget.

## 7. Sortie attendue

- Spectromètre entraîné pour chaque λ_budget exploré.
- Courbe Pareto λ_budget → (qualité, complexité effective).
- Diagramme de Phase final.
- Évidence d'alignement R_target ↔ structure réelle (préparation du test d'Élasticité phase 5).

## 8. Critères go/no-go

**Go** :
- Courbe Pareto (mesurée *après* 4b, donc sans distillation) avec une portion strictement dominante par rapport au Backbone seul et au Transformer plein.
- Diagramme de Phase croissant : R_target augmente avec le stress imposé.
- Au moins une valeur de λ_budget pour laquelle le Spectromètre n'est ni saturé (R_target ≈ R_max partout) ni éteint (R_target ≈ 0 partout).
- Dérive 4a → 4b bornée (variation < 30% à la bascule).

**No-go** :
- Courbe Pareto plate après 4b → un seul régime déguisé, ou la phase 4b a effacé tout l'apprentissage de 4a.
- Diagramme de Phase plat → le Spectromètre n'a pas appris à différencier les régimes.
- R_target décorrélé du stress local construit → le Spectromètre a appris un proxy non causal (position absolue, statistique globale).
- Échec de la transition 4a → 4b (pas de convergence de L_distillation, ou corrélation Spearman < 0,80) → diagnostic du signal de surprise (retour Phase 1.5).

## 9. Pièges à éviter

- **Gel précoce** — λ_budget trop grand trop tôt → Spectromètre s'éteint définitivement. **Mitigation** : warmup avec λ_budget = 0 sur les premières steps de l'étage 1, montée progressive (schedule pré-enregistré).
- **Bypass dégénéré** — le Backbone porte tout, l'extension ne corrige qu'un offset constant. **Test révélateur** : geler le Backbone, vérifier que l'extension porte une *fonction* du contexte, pas un biais.
- **Métrique de complexité malhonnête** — toujours compter les FLOPs *effectifs* (gates allumés × dimensions), jamais les FLOPs nominaux (R_max). Sans cette honnêteté, le polymorphisme est invisible.
- **Spectromètre sur-paramétré** — capacité élevée → encode des heuristiques de tâche → triche. Le test d'Identifiabilité (phase 5) le débusquera, mais autant éviter d'en arriver là.

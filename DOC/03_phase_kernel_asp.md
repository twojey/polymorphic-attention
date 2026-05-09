# 03 — Phase 3 : Synthèse — l'ASPLayer

C'est le passage de la théorie au code. On construit l'**ASPLayer** (alias *Adaptive Structural Processor*) : une unité de calcul capable de modifier sa propre capacité au vol — son rang — sans interrompre le flux des données ni casser la dérivabilité.

## 1. Architecture duale : Backbone dérivé + Extension

L'ASPLayer hybride deux régimes :

- **Backbone (régime stationnaire)** — opérateur structuré paramétré, **dérivé du dictionnaire SCH** produit en phase 2. Sa structure (Toeplitz, Hankel, Cauchy, combinaison de ces classes, ou autre primitive révélée par l'audit) est *imposée par la mesure*, pas importée d'une famille existante (pas de Mamba2, pas de S4 préfabriqué). Ses paramètres sont appris par descente de gradient. Coût : O(N), mémoire bornée.
- **Extension low-rank Matriochka (régime dynamique)** — branche activée quand le stress structurel augmente au-delà de ce que le Backbone capte. Décomposition U·Vᵀ avec rang r_t ∈ [r_min, R_max] modulé en temps réel par le Spectromètre. r_min est le plancher du Spectromètre (typiquement 0).

Composition :

```
y_t = Backbone(x_t) + ΔAttn(x_t ; r_t)
```

L'addition (et non la substitution) est essentielle : si le Spectromètre éteint l'extension (r_t = 0), l'ASPLayer dégénère proprement vers le Backbone seul. Si l'extension est pleinement ouverte (r_t = R_max), elle apporte la capacité nécessaire pour atteindre la qualité de l'Oracle.

### 1.1 Dérivation du Backbone depuis le dictionnaire SCH

Le dictionnaire SCH (livrable phase 2) liste, pour chaque régime de stress, la classe d'opérateur structuré qui fitte le mieux la matrice d'attention de l'Oracle. Schéma de dérivation :

| Si phase 2 révèle (régimes faciles) | Alors Backbone = |
|-------------------------------------|------------------|
| Toeplitz dominant, rang 1–2 | Convolution causale longue paramétrée `y_t = Σ_k h_k · x_{t-k}`, h_k appris |
| Hankel dominant, rang r petit | SSM générique `h_{t+1} = A h_t + B x_t, y_t = C h_t`, A, B, C appris **sans imposer HiPPO ni selectivity** |
| Cauchy dominant | Opérateur d'interpolation rationnelle paramétré (poles appris) |
| Vandermonde dominant | Opérateur polynomial paramétré |
| Combinaison de classes | Somme ou composition des opérateurs ci-dessus |

**Important** : ces classes ressemblent à des composants de Mamba, S4, Hyena, etc. — c'est normal, ces architectures ont historiquement été *partiellement* dérivées des mêmes mathématiques. Ce qu'on évite, c'est d'**hériter du package complet** d'une de ces architectures (avec ses choix d'init spécifique, ses contraintes de paramétrage non motivées par notre mesure, ses optimisations ad-hoc). On ne reprend que la **forme structurelle** que la phase 2 a justifiée.

### 1.2 Multi-domaine

Si la phase 2 a révélé que la SCH est domain-spécifique (cf. DOC/02 section 4b), le Backbone peut être :
- **Configurable par domaine** (paramétrage différent selon le domaine cible),
- ou **universel** mais avec un mélange interne de primitives, dont les poids relatifs sont appris par domaine.

Choix instruit par les données phase 2, pas a priori.

## 2. Hiérarchie Matriochka des bases U et V

C'est le point technique critique. Pour que le modèle puisse passer d'un rang 4 à un rang 32 *de manière fluide*, les bases U et V doivent être **imbriquées** :

- Une seule grande matrice U_max ∈ ℝ^{d × R_max} et une seule V_max ∈ ℝ^{d × R_max}.
- Un masque dynamique m_t ∈ [0, 1]^{R_max} émis par le Spectromètre, **monotone décroissant** : m_{t,1} ≥ m_{t,2} ≥ … ≥ m_{t,R_max}.
- À l'instant t, la correction effective est :

```
ΔAttn_t(x) = (U_max ⊙ m_t) (V_max ⊙ m_t)ᵀ · x
```

(produit composante par composante sur les colonnes).

**Conséquence structurelle.** Les premières colonnes de U et V apprennent les structures les plus générales (toujours utiles) ; les colonnes ultérieures apprennent des raffinements progressifs. Augmenter r_t = r → r+1 *ajoute* une colonne, ne *remplace* aucune. La continuité du comportement est garantie par construction.

Cette hiérarchie est entraînée par une **loss Matriochka** appliquée sur la **sortie** de l'ASPLayer (et non sur la matrice U_max V_maxᵀ). Pour une entrée x avec cible y_target :

```
L_matriochka = Σ_{r ∈ S} w_r · L_task( ASPLayer(x; mask = [1×r, 0×(R_max−r)]) , y_target )
```

où S est un sous-ensemble échantillonné de {1, …, R_max} pendant l'entraînement, w_r pondère l'importance de chaque rang, et `mask = [1×r, 0×(R_max−r)]` désigne un masque idéal qui n'active que les r premières colonnes de U_max et V_max.

**Pourquoi sur la sortie, pas sur la matrice.** U_max et V_max sont des paramètres **statiques** (un seul jeu par couche). Une formulation matrice-cible `‖A* − U_max[:,:r] V_max[:,:r]ᵀ‖²` demanderait à une matrice fixe d'approximer des A* qui varient avec l'entrée — non résoluble sans dépendance à x. La formulation output-based contourne ce problème : la même paire (U_max, V_max) génère des sorties différentes pour différentes entrées via le calcul `(U_max ⊙ m_t)(V_max ⊙ m_t)ᵀ x`. La loss demande seulement que ces sorties soient bonnes pour la tâche à chaque rang de troncature.

Inspiration directe : *Matryoshka Representation Learning* (Kusupati et al.) — eux appliquaient la loss à des embeddings tronqués, on l'applique à des sorties d'ASPLayer tronquées en rang.

**Cohérence avec L_consistency.** Les deux losses opèrent sur le même objet (la sortie de l'ASPLayer). L_matriochka demande que chaque sortie tronquée soit *bonne* (proche de y_target). L_consistency demande que les sorties à rangs voisins soient *proches l'une de l'autre*. Ensemble, elles produisent une famille de sorties qui dégradent monotonement et lissement quand r diminue.

### 2.1 Le terme de Consistency (V2)

La loss Matriochka seule ne garantit pas que les sorties du modèle varient *lissement* avec α_t. Le Soft-Mask (section 4.1) a une pente qui dépend de α, et combinée à la loss Matriochka, elle peut produire des discontinuités fonctionnelles entre rangs voisins — la qualité saute brutalement entre r=k et r=k+1 au lieu de monter continûment. Ce comportement est exactement ce que le Spectromètre (phase 4) ne pourra pas exploiter.

On ajoute un **terme de Consistency** qui pénalise la variation de la sortie entre rangs voisins, sur la même entrée :

```
L_consistency = E_{r ~ U(S), δ ~ U{−1, +1}} [ ‖y(x ; r) − y(x ; r+δ)‖² ]
```

où y(x ; r) est la sortie de l'ASPLayer en imposant un masque idéal `[1,…,1,0,…,0]` à r colonnes activées. L'espérance est calculée sur le mini-batch en pratique (un échantillon (r, δ) par exemple suffit).

La loss totale en phase 3 devient :

```
L_phase3 = L_task + λ_M · L_matriochka + λ_C · L_consistency
```

avec λ_C calibré pour que la pente de la qualité en fonction de r soit *monotone et lisse* (sanity check de la section 6 enrichie : on vérifie en plus l'absence de sauts).

**Effet attendu.** Sans Consistency, le Spectromètre apprend dans un paysage où passer de r=k à r=k+1 peut faire chuter ou exploser la qualité de manière imprévisible — l'optimisation est instable. Avec Consistency, l'augmentation de r se traduit par un gain de qualité graduel, ce qui permet au Spectromètre de calibrer son α_t par descente de gradient stable.

## 3. Forward Pass — algorithme

Pour un token x_t en position t :

1. **Projection latente** — x_t → représentation latente z_t (typiquement une LayerNorm + projection linéaire).
2. **Diagnostic** — le Spectromètre (cf. phase 4) reçoit un signal s_t (résidu prédictif local du Backbone, KL local, ou norme de gradient en train) et émet un score de rang scalaire α_t ∈ [0, 1] (ou un vecteur de gates m_t directement).
3. **Expansion** — projection vers le rang entier cible :
   ```
   R_target = round(α_t · R_max)        (forward)
   ```
   La continuité du gradient sur cette opération est traitée à la section 4.
4. **Calcul polymorphe** — assemblage des deux branches :
   ```
   y_t = Backbone(x_t) + Σ_{i=1..R_target} U_max[:, i] · V_max[:, i]ᵀ · x_t
   ```
5. **Normalisation** — passage par une LayerNorm adaptée à la variance variable induite par le rang dynamique (LayerNorm conditionnée par r_t, ou simple LayerNorm post-addition selon ablation).

## 4. Continuité du gradient — Soft-Masking

L'opération `round(α · R_max)` n'est pas dérivable. Sans précaution, le gradient du Spectromètre est nul et l'allocation n'apprend rien. Solutions, du plus simple au plus robuste :

### 4.1 Soft-Masking (recommandé en première intention)

Au lieu de couper brutalement à R_target, appliquer une fonction de masquage **douce** :

```
m_{t,i} = σ( β · (α_t · R_max − i + ½) )
```

avec β > 0 (raideur). Quand β → ∞, on retrouve un masque binaire ; quand β est modéré, le masque décroît progressivement autour de l'indice cible. Le gradient « sent » les rangs supérieurs au seuil et apprend quand les activer.

Cette forme respecte automatiquement la **monotonie décroissante** exigée par la hiérarchie Matriochka : plus i est grand, plus m_{t,i} est petit.

### 4.2 Straight-Through Estimator (STE)

Forward avec valeur arrondie, backward avec gradient continu :

```
r_discrete = round(r_continuous)
r_final    = r_continuous + (r_discrete − r_continuous).detach()
```

À combiner avec le Soft-Masking si on veut le meilleur des deux mondes : forward propre, backward dérivable, comportement raide en évaluation.

### 4.3 Gumbel-Softmax

Échantillonnage différentiable du rang avec température τ. Réservé aux cas où on veut explicitement de la stochasticité (exploration). Coût d'optimisation plus élevé, à ne tenter qu'après échec documenté du Soft-Masking.

## 5. Initialisation des bases U_max, V_max

Les bases sont des paramètres statiques : leur état initial influence la trajectoire d'optimisation (Matriochka + Consistency + L_task) mais pas la forme finale des sorties. Deux stratégies, **avec random comme défaut**.

### 5.1 Init aléatoire (défaut)

Init standard (Xavier uniform, ou orthogonale par bloc R_max). Aucune information venue de l'Oracle n'est injectée.

**Pourquoi par défaut.** Les sanity checks (saturation, effondrement, monotonie, lissité) doivent passer dans cette configuration **avant** toute considération de smart init. Sinon on confond un succès architectural avec un succès d'initialisation, ce qui contamine le test 5d (Pareto) face aux baselines qui n'ont pas accès à l'Oracle.

### 5.2 Smart init Matriochka (ablation)

Si la version random passe et qu'on cherche à accélérer la convergence ou à améliorer le plafond, on peut initialiser les premières colonnes de U_max, V_max à partir du **Diagnostic de Spécialisation des têtes** produit en phase 2 (DOC/02 section 5b).

**Schéma**.

1. Récupérer la liste des K têtes les plus spécialisées de l'Oracle (haute spec_h = haute variance de r_eff_h sur les régimes).
2. Pour chacune, extraire les matrices d'attention A_h **sur le set `init_phase3`** (cf. règle des trois sets disjoints, DOC/01 section 8.6 — set distinct du training Oracle ET de l'audit SVD).
3. SVD : A_h = U_h Σ_h V_hᵀ. Garder les top-k_h vecteurs singuliers, k_h ∝ r_eff_h moyen sur les régimes durs.
4. Concaténer dans U_max[:, 0:K_total] et V_max[:, 0:K_total], ordonnés par valeur singulière décroissante.
5. Le reste U_max[:, K_total:R_max], V_max[:, K_total:R_max] : init aléatoire.

**Cohérence Matriochka.** Les premières colonnes (toujours actives quel que soit r_t) portent la structure la plus saturée d'information — alignée avec le principe de la hiérarchie Matriochka.

### 5.3 Garde-fous méthodologiques (smart init)

- **Trois sets disjoints obligatoires** : `train_oracle` (phase 1), `audit_svd` (phase 2), `init_phase3` (cette init). Sans cette tri-partition, on injecte de l'information du test set vers l'init de l'ASPLayer.
- **Pas de back-propagation vers l'init.** Les vecteurs singuliers extraits de l'Oracle sont *figés* lors de l'extraction. Pas d'optimisation jointe Oracle ↔ ASP.
- **Random init doit avoir passé.** Si on n'a pas validé les sanity checks en random, smart init n'est pas une option — elle masquerait un défaut architectural.
- **Reporting honnête en phase 5.** Si smart init est utilisé pour les chiffres principaux, l'ablation random doit aussi être reportée, et la différence quantifiée. Sinon on compare un ASP "boosté" à des baselines "vanilla".
- **Pré-enregistrement.** La stratégie d'init (random ou smart, et quels K et k_h pour smart) est arrêtée dans `OPS/configs/phase3/init.yaml` *avant* l'entraînement.

### 5.4 Information secondaire — convergence et architecture

Si smart init améliore beaucoup la qualité finale, c'est un signal que l'optimisation Matriochka + Consistency est fragile et mérite mieux (schedulers, warm-up). Si smart init n'améliore rien, c'est un résultat plus fort : l'architecture s'auto-organise depuis n'importe quelle init raisonnable.

## 6. Spécification technique (boilerplate)

Squelette de référence pour l'implémentation (les noms exacts dépendent de la stack arrêtée à la fin de phase 1) :

| Composant | Type structurel | Rôle |
|-----------|-----------------|------|
| `W_base` (Backbone) | Opérateur structuré dérivé du dictionnaire SCH (Toeplitz, Hankel, Cauchy, ou combinaison ; cf. section 1.1). Pas une famille pré-importée. | Régime stationnaire, capacité dictée par la phase 2 |
| `U_mat` | paramètre `(d, R_max)` | Réserve de capacité relationnelle, base gauche Matriochka |
| `V_mat` | paramètre `(d, R_max)` | Réserve de capacité relationnelle, base droite Matriochka |
| `Spectromètre` | mini-MLP `(d → 1)` ou `(d → R_max)` | Prédit le stress local α_t ou le vecteur m_t |
| `Soft-Mask` | fonction (α, R_max, β) → m | Génère le masque doux monotone décroissant |
| `Rank_Sampler` (optionnel) | Gumbel-Softmax | Variante stochastique du masquage |
| `LayerNorm_r` | LayerNorm | Normalisation post-addition, éventuellement conditionnée par r_t |

## 7. Sanity checks de la phase

Trois vérifications obligatoires avant de fermer la phase :

1. **Saturation** — à m_t = 1 ∀t (extension pleinement ouverte), l'ASPLayer atteint la borne de qualité de l'attention pleine sur Structure-MNIST. Sinon, R_max est sous-dimensionné par rapport à la cible de phase 2.
2. **Effondrement** — à m_t = 0 ∀t, l'ASPLayer ≡ Backbone seul. Sanity check de l'addition.
3. **Monotonie Matriochka** — pour un masque idéal `[1,…,1,0,…,0]` à r_t croissant, la qualité décroît *monotonement* avec la baisse de r_t. Si la qualité oscille, la loss Matriochka ne fait pas son travail et la phase 4 sera instable.
4. **Lissité (Consistency)** — la pente de la qualité en fonction de r ne présente pas de saut. Quantitativement : la dérivée seconde discrète |q(r+1) − 2·q(r) + q(r−1)| reste sous un seuil pré-enregistré pour tout r ∈ [r_min, R_max−1]. Si non, augmenter λ_C dans la loss Consistency.

## 8. Sortie attendue de la phase

- Implémentation de référence de l'ASPLayer.
- Stratégie d'init pré-enregistrée (random par défaut ; smart init en ablation si demandée).
- Quatre sanity checks documentés (saturation, effondrement, monotonie, lissité), passés en *random init*.
- Si smart init est exécuté : ablation random vs smart documentée avec écart quantifié sur la qualité finale et la vitesse de convergence.
- Visualisation : à entrée connue, heatmap de R_target apprise (idéalement *avant* phase 4, en imposant R_target = r_target prédit par la loi de transfert phase 2 — sert de baseline distillée).
- Recommandation explicite : valeur de β (raideur du Soft-Mask), λ_M (poids Matriochka), λ_C (poids Consistency), choix de la stratégie d'échantillonnage S, choix de la LayerNorm.

## 9. Critère go/no-go

**Go** : les quatre sanity checks sont passés. À R_max calibré sur la cible de phase 2, l'ASPLayer atteint la borne phase 1.

**No-go** :
- Plafond strictement sous la borne phase 1 même à m_t = 1 → expressivité insuffisante. Revoir le Backbone ou élargir R_max.
- Décroissance non monotone avec r_t → la loss Matriochka ne fait pas son travail. Revoir l'échantillonnage S et la pondération w_r avant de lancer phase 4.
- Sauts de qualité entre rangs voisins (lissité KO) → augmenter λ_C ; si insuffisant, revoir la formulation du Soft-Mask.

## 10. Question ouverte

Faut-il une **seule** correction Matriochka globale ou **plusieurs têtes** indépendantes ? Le produit hiérarchie Matriochka × multi-têtes est non trivial : l'imbrication peut s'appliquer entre têtes, à l'intérieur de chaque tête, ou les deux. À instruire empiriquement, pas par analogie.

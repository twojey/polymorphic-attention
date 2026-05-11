# 01b — Phase 1.5 : Calibration de l'Indicateur de Stress

> **Cadrage** (clarifié 2026-05-11) : cette phase fait partie de la **Partie 2 — Validation hypothèse polymorphique (ASP)**. Elle teste 3 signaux candidats spécifiques (S_KL, S_Grad, S_Spectral) qui sont un **sous-ensemble** du catalogue exhaustif de la **Partie 1 — Science fondamentale** ([00b_classification_proprietes.md](00b_classification_proprietes.md), catégories A1, B2, C1, C6). Un échec ici n'invalide pas le projet sub-quadratique au sens large : il ferme la voie ASP-via-allocation-dynamique mais d'autres approches restent ouvertes (kernel approx, sparse, low-rank, state-space). Cf. également ROADMAP.md "Stage 1.5+ — Classification mathématique étendue".

**Identification Gate (Partie 2).** Cette phase intercalaire entre la métrologie (phase 1) et l'audit spectral (phase 2) répond à *la* question dont dépend la suite **du protocole ASP** : existe-t-il un signal observable token par token qui corrèle avec le stress structurel **et** reste insensible au bruit pur ?

Si la réponse est non, le Spectromètre (phase 4) ne peut pas exister. La construction de l'ASPLayer (phase 3) elle-même devient sans objet : on aurait une architecture capable d'allouer dynamiquement du rang, mais aucun signal pour piloter cette allocation. **La voie ASP s'arrête** (mais pas le projet sub-quadratique global — pivot possible vers d'autres approches).

C'est pourquoi cette phase est un **gate de la Partie 2** : aucun signal validé → aucun passage en phase 2 ASP.

## 1. Le trio de candidats

| Symbole | Signal | Régime de calcul | Coût d'inférence |
|---------|--------|------------------|------------------|
| **S_KL** | KL local entre p(x_t \| x_<t) et un baseline empirique global | Inférence | O(V), V = vocabulaire |
| **S_Grad** | Norme du gradient local par rapport à L_task | Train uniquement | O(coût backward) |
| **S_Spectral** | Rang effectif r_eff de la sub-matrice d'attention sur fenêtre glissante K | Inférence | O(K² log K) via SVD partielle |

S_KL est un proxy *psychologique* (réaction du modèle), S_Grad et S_Spectral sont des mesures *physiques* (du calcul).

**Hypothèse de travail (issue de l'analyse théorique).** S_Grad est vraisemblablement non-identifiable (asymétrie noise/structure) ; S_KL et S_Spectral sont les vrais finalistes. Mais cette hypothèse est elle-même testée empiriquement — on les soumet tous les trois au même banc d'essai.

### 1.1 Pourquoi S_Surprise a été retiré

Une version antérieure du protocole listait un quatrième candidat, `S_Surprise = ‖y_t^{Backbone} − y_t‖ / ‖y_t‖` — résidu prédictif d'un Backbone proxy. Il a été retiré pour deux raisons :

1. **Importation interdite** — calculer S_Surprise exige d'instancier un Backbone (SSM mince) avant la phase 3. Or le principe Discovery > Reproduction (DOC/00 section 4b) interdit d'importer une famille d'architecture (Mamba, S4, etc.) avant que la phase 2 n'ait dérivé la structure du Backbone.
2. **Asymétrie noise/structure** — même au-delà de l'argument méthodologique, S_Surprise échouerait probablement le test d'immunité au bruit : sur du bruit blanc, le modèle est aussi en erreur, le résidu fire pareil. Pas de discrimination structure / désordre.

Le retrait n'est pas une perte : un signal qui aurait échoué de toute manière, en plus de coûter une importation architecturale prématurée.

## 2. Le banc de test

### 2.1 Dataset hybride

50 % de séquences SSG à structure variable :
- ω ∈ {2, 4, 6, 8, 10, 12} (récursion)
- Δ ∈ {64, 256, 1024, 4096} (distance)
- ℋ basse (peu de bruit)

50 % de séquences à structure absente :
- (ω, Δ) = (0, 0)
- ℋ étalée sur toute sa plage

Les deux moitiés sont **mélangées par séquence** (pas par batch), pour que chaque token soit évalué dans son propre contexte.

### 2.2 Mesure

Pour chaque token t de chaque séquence :
1. Calculer les trois signaux S_KL(t), S_Grad(t), S_Spectral(t).
2. Enregistrer les paramètres de stress du token : ω(t), Δ(t), ℋ(t).
3. Appliquer l'agrégation cross-layer/cross-head (section 3).

Sur l'ensemble des paires (signal, paramètre), calculer les **corrélations de Spearman** :

```
ρ_{S, ω} = Spearman( {S(t)}_t , {ω(t)}_t )
ρ_{S, Δ} = Spearman( {S(t)}_t , {Δ(t)}_t )
ρ_{S, ℋ} = Spearman( {S(t)}_t , {ℋ(t)}_t )
```

Spearman et non Pearson : robustesse aux relations monotones non linéaires.

## 3. Agrégation cross-layer / cross-head

S_KL et S_Spectral se calculent par couche, par tête. Le Spectromètre prend **un signal par token**. La réduction est :

```
1.  Per-layer max-pool sur les têtes : s_layer(ℓ, t) = max_h S(ℓ, h, t)
2.  Concatenate sur la deep-stack    : s(t) = [ s_layer(1, t) ; s_layer(2, t) ; … ; s_layer(L, t) ]
```

**Justification du max.** Une seule tête qui détecte un pic structurel suffit à justifier l'allocation de rang. Une moyenne diluerait ce signal sous la masse des têtes inactives.

**Justification de la concaténation.** La progression du stress à travers la profondeur est une signature plus riche qu'une seule valeur agrégée. Le Spectromètre voit la *trajectoire* couche par couche, pas seulement un résumé.

Sortie : un vecteur de dimension L par signal, soit 4L valeurs par token au total. Pour un Backbone à 6 couches, 24 valeurs scalaires en entrée du Spectromètre. Coût négligeable.

## 4. Critères de sélection

Pour qu'un signal S soit retenu pour la phase 4, il doit valider :

```
max( ρ_{S, ω}, ρ_{S, Δ} ) > 0.70    (Sensibilité Structurelle)
|ρ_{S, ℋ}| < 0.20                    (Immunité au Bruit)
```

### 4.1 Justification du seuil 0.70 (Dominance de l'Information)

ρ = 0.70 ↔ R² ≈ 0.49. Le signal explique au moins la moitié de la variance du stress structurel. En deçà, le résidu inexpliqué domine et la supervision du Spectromètre est noyée dans le bruit. C'est le minimum requis pour qu'un signal soit *informatif* à l'échelle d'un token.

### 4.2 Justification du seuil 0.20 (Innocuité Stochastique)

|ρ| < 0.20 ↔ R² < 0.04. Le bruit blanc ne peut pas expliquer plus de 4 % de la variance de R_target via ce signal. C'est la ceinture de sécurité contre l'inflation du compute en déploiement : un Spectromètre alimenté par ce signal ne peut pas dépenser plus de 4 % de son budget de rang à cause du bruit pur.

### 4.3 Multi-signal autorisé

On n'exige pas un signal unique qui passe sur les deux axes (ω et Δ). Si S_KL passe sur ω mais pas sur Δ, et S_Spectral passe sur Δ mais pas sur ω, **les deux** sont retenus et le Spectromètre les reçoit en entrée. L'allocation de rang devient une décision multi-factorielle.

Critère de retenue : au moins un signal doit passer le seuil 0.70 sur **au moins un axe** ; tous les signaux retenus doivent passer le seuil 0.20 sur ℋ.

## 5. Phase 1.5b — Hypothèse de Distillabilité

Si S_Spectral est retenu, son coût d'inférence (SVD partielle par token) peut être prohibitif en production. On teste alors une **hypothèse secondaire** :

> Un MLP léger peut apprendre à prédire S_Spectral à partir de x_t (et éventuellement d'un état récurrent court), avec une erreur acceptable.

Si cette hypothèse passe, on déploie le MLP léger comme proxy au lieu de la SVD directe (le "teacher" est S_Spectral pendant l'entraînement, le "student" est le MLP en déploiement).

### Protocole

1. Calculer S_Spectral sur tout le banc de test (entraînement + validation).
2. Entraîner un MLP léger M(x_t) → S_Spectral(t) sur l'entraînement.
3. Mesurer MSE et corrélation Spearman entre M(x_t) et S_Spectral(t) sur le set held-out.

### Critère

```
ρ_Spearman( M(x), S_Spectral ) > 0.85
ET   MSE relative < seuil pré-enregistré
```

Le seuil ρ > 0.85 ici est plus exigeant que le 0.70 de la section 4 : un *student* qui imite un *teacher* doit le faire mieux qu'une régression sur le stress brut, sinon on perd l'information que le teacher avait à offrir.

### Cas d'échec

Si le MLP n'arrive pas à approximer S_Spectral, deux fallbacks :

1. **Acceptation du coût** — utiliser S_Spectral en direct, payer le coût SVD à l'inférence. Acceptable si le coût total reste sous le budget Pareto cible.
2. **Simplification du Backbone** — réduire la dimension d'attention pour rendre la SVD partielle moins coûteuse, au prix d'une légère baisse de qualité.

Le choix entre ces deux fallbacks est documenté ; il n'invalide pas la phase 1.5 globalement.

## 6. Sortie attendue

Un rapport contenant :

1. **Matrice de corrélation 3 × 3** : (S_KL, S_Grad, S_Spectral) × (ω, Δ, ℋ), avec intervalles de confiance bootstrap. Optionnellement étendue à `(3 × 3) × N_domaines` si plusieurs Oracles (multi-Oracle, cf. DOC/01 section 7b) sont calibrés.
2. **Liste des signaux retenus** pour la phase 4, avec axes de stress couverts par chacun.
3. **Verdict de Distillabilité** (si S_Spectral est retenu) : ρ_student/teacher, MSE, fallback choisi le cas échéant.
4. **Choix d'agrégation finalisé** : confirmation de Max-Pool layer + Concat deep-stack, ou révision si le banc révèle un meilleur schéma.

## 7. Critère go/no-go

**Go** : au moins un signal passe les deux conditions de la section 4 (sensibilité + innocuité), sur au moins un axe de stress. Si S_Spectral est dans la liste, sa Distillabilité est validée ou un fallback est choisi.

**No-go** : aucun signal ne passe les deux conditions, sur aucun axe. Conclusion : *l'allocation dynamique de rang par token est, dans le cadre de ce SSG et de cet Oracle, une illusion statistique*. On ne peut pas piloter ce qu'on ne peut pas observer. Le protocole s'arrête.

C'est exactement le cas où la falsifiabilité paie son coût : on apprend une chose vraie (négatif) plutôt qu'on construit une chose fausse (positif fragile).

## 8. Pièges méthodologiques

- **Calibration du baseline KL avant le test.** Le baseline empirique global de S_KL doit être calibré une fois pour toutes sur un échantillon stationnaire de bruit, *avant* le banc d'essai. Le calibrer après risque d'optimiser le baseline pour passer le test — fuite méthodologique majeure.
- **Choix de la fenêtre K pour S_Spectral.** Pré-enregistrer K (typiquement 32, 64, ou 128). Tester plusieurs K *après* le verdict est de l'optimisation post-hoc.
- **Spearman demande des échantillons indépendants.** Pour éviter de surestimer ρ, sous-échantillonner les paires (un token tous les K, par exemple). Pré-enregistrer la stratégie de sous-échantillonnage.
- **Bootstrap pour les intervalles de confiance.** Les seuils 0.70 et 0.20 sont des valeurs ponctuelles ; rapporter avec IC95 % bootstrap pour éviter les conclusions hâtives sur des différences non significatives.
- **S_Grad uniquement à l'entraînement.** Si on retient S_Grad pour entraîner le Spectromètre mais qu'il n'est pas disponible à l'inférence, l'inférence devient inconsistante avec l'entraînement. Soit on exclut S_Grad de la liste finale, soit on apprend un proxy de S_Grad analogue à la Distillabilité de S_Spectral.

## 9. Position dans le protocole

```
Phase 1 (RCP / SSG)              →  Oracle entraîné, cartes de stress
       ↓
Phase 1.5 (Calibration Signal)   →  Signal(s) de stress validé(s), Identification Gate
       ↓
Phase 2 (Audit Spectral)         →  SCH corroborée, loi de transfert
       ↓
Phase 3, 4, 5                    →  …
```

Phase 1.5 utilise les **mêmes données** que la phase 1 (Structure-MNIST via le SSG, et tout autre domaine consommé par les Oracles supplémentaires) et les **mêmes Oracles**. Pas d'entraînement supplémentaire : on instrumente les Oracles existants pour calculer les trois signaux candidats. Coût additionnel : quelques heures de SVD partielle batchée + une régression Spearman par domaine.

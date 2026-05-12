"""Famille U — Sparse-structurées (DOC/CATALOGUE §U).

U1 Butterfly distance, U2 Monarch distance, U3 Pixelfly, U4 Block-sparse,
U5 Sparse + low-rank.

V1 : distance Frobenius à la classe via mask (pattern de support) ou
décomposition itérative simple. Pour Butterfly/Monarch, le V1 mesure
la concentration de l'énergie sur le pattern de support théorique,
plutôt que de factoriser ALS (V2 lourd).

Famille CRITIQUE pour ASP : si une matrice d'attention est ε-proche de
Butterfly ou Monarch, on peut la remplacer par cette structure en
O(N log N) ou O(N√N).
"""

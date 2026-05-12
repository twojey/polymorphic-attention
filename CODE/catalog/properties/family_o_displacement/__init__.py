"""Famille O — Rangs de déplacement (Kailath / Pan).

Spec : DOC/00b §O.

L'opérateur de déplacement ∇_Z(A) = A − Z·A·Zᵀ (avec Z shift down) annule
exactement les matrices Toeplitz. Plus généralement, les classes de matrices
structurées admettent un rang de déplacement ≤ 1 ou 2 pour un opérateur
∇ approprié — c'est l'invariant théorique propre du catalogue.

V1 : O1 Toeplitz-like seulement. O2 Cauchy-like et O3 Vandermonde-like
nécessitent des paires (D_x, D_y) paramétrées — relevés à V2 quand on
aura les Projectors Cauchy/Vandermonde.
"""

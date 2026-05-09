"""
s_grad.py — signal S_Grad : norme du gradient local par rapport à L_task.

Spec : DOC/01b §1.

Coût : O(coût backward). Calculable uniquement à l'entraînement.

Convention V1 : pour chaque token t, calculer ‖∂L/∂x_t‖₂ où x_t est
l'embedding (post tok+pos) du token t, et L est la cross-entropy de
prédiction du QUERY token. Cette quantité capture la "sensibilité" de
la loss à chaque position d'entrée.

Limite : S_Grad n'est pas calculable à l'inférence (DOC/01b §8). Si retenu,
exige un proxy distillé analogue à la Distillabilité de S_Spectral.
"""

from __future__ import annotations

import torch
from torch import nn


def compute_s_grad(
    *,
    model: nn.Module,
    tokens: torch.Tensor,
    query_pos: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """S_Grad par (batch, token).

    Retourne (B, N) FP64. Le modèle doit avoir un attribut `tok_embed` et `pos_embed`.
    """
    model.eval()  # pas de dropout pour le gradient de mesure
    for p in model.parameters():
        p.requires_grad_(False)

    embeddings = model.tok_embed(tokens) + model.pos_embed[: tokens.size(1)].unsqueeze(0)
    embeddings = embeddings.detach().requires_grad_(True)

    # Forward "shortcuté" : on ré-injecte directement les embeddings.
    # Pour V1, on suppose que `model.forward` n'est pas modifiable mais
    # on peut appeler les blocs manuellement.
    x = embeddings
    pad_mask = tokens == model.cfg.pad_id
    attn_mask = torch.zeros(tokens.size(0), 1, 1, tokens.size(1), device=tokens.device, dtype=x.dtype)
    attn_mask = attn_mask.masked_fill(pad_mask[:, None, None, :], float("-inf"))

    for block in model.blocks:
        x = block(x, attn_mask=attn_mask, capture_attn=False)
    x = model.ln_f(x)

    idx = query_pos[:, None, None].expand(-1, 1, x.size(-1))
    h = x.gather(dim=1, index=idx).squeeze(1)
    logits = model.head(h)
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")

    grads = torch.autograd.grad(loss, embeddings, create_graph=False, retain_graph=False)[0]
    # ‖grad‖₂ par token
    s = grads.to(torch.float64).norm(dim=-1)  # (B, N)

    # Reset requires_grad pour les paramètres du modèle
    for p in model.parameters():
        p.requires_grad_(True)

    return s

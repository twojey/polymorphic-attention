"""
distillability.py — sous-phase 1.5b. Test de la Distillabilité de S_Spectral.

Spec : DOC/01b §5.

Hypothèse : un MLP léger M(x_t) peut prédire S_Spectral(t) à partir de
l'embedding x_t (et éventuellement d'un état récurrent court).

Critère : ρ_Spearman(M(x), S_Spectral) > 0.85 ET MSE relative < seuil.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats
from torch import nn


class StudentMLP(nn.Module):
    """MLP léger : (B, N, d_model) -> (B, N) prédiction scalaire de S_Spectral.

    Optionnellement consomme un état récurrent court (concaténé en input).
    """

    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class DistillabilityResult:
    rho_spearman: float
    mse: float
    mse_relative: float
    n_train: int
    n_eval: int
    passed: bool


def train_student(
    *,
    embeddings_train: torch.Tensor,    # (N, d) flatten tokens
    targets_train: torch.Tensor,       # (N,) S_Spectral teacher
    embeddings_eval: torch.Tensor,
    targets_eval: torch.Tensor,
    d_model: int,
    hidden: int = 128,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    rho_threshold: float = 0.85,
    mse_relative_threshold: float = 0.5,
    device: torch.device | None = None,
) -> DistillabilityResult:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = StudentMLP(d_model=d_model, hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    embeddings_train = embeddings_train.to(device)
    targets_train = targets_train.to(device).float()
    embeddings_eval = embeddings_eval.to(device)
    targets_eval = targets_eval.to(device).float()

    n_train = embeddings_train.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            x = embeddings_train[idx]
            y = targets_train[idx]
            pred = student(x.unsqueeze(0)).squeeze(0)
            loss = nn.functional.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    student.eval()
    with torch.no_grad():
        pred_eval = student(embeddings_eval.unsqueeze(0)).squeeze(0)
    pred_np = pred_eval.cpu().numpy()
    target_np = targets_eval.cpu().numpy()
    rho = float(stats.spearmanr(pred_np, target_np).statistic)
    mse = float(np.mean((pred_np - target_np) ** 2))
    target_var = float(np.var(target_np)) + 1e-12
    mse_rel = mse / target_var
    return DistillabilityResult(
        rho_spearman=rho,
        mse=mse,
        mse_relative=mse_rel,
        n_train=n_train,
        n_eval=embeddings_eval.size(0),
        passed=(rho > rho_threshold) and (mse_rel < mse_relative_threshold),
    )

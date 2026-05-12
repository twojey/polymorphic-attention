"""
backbone_concrete.py — Backbones concrets dérivés du dictionnaire SCH phase 2.

Spec : DOC/03 §1.1.

Trois implémentations selon la classe dominante identifiée en phase 2 :

- `ToeplitzConvBackbone` : convolution causale longue paramétrée
    y_t = Σ_k h_k · x_{t-k}, h_k appris (FFT-based pour O(N log N))
- `HankelSSMBackbone` : SSM générique sans HiPPO/selectivity
    h_{t+1} = A h_t + B x_t, y_t = C h_t (A, B, C appris)
- `CompositeBackbone` : somme additive de plusieurs primitives

Le choix concret est ARRÊTÉ par lecture du rapport phase 2 (cf.
`Phase2Verdict.class_dominante` dans run_train.py).
"""

from __future__ import annotations

import torch
from torch import nn

from phase3_kernel_asp.backbone import Backbone


class ToeplitzConvBackbone(Backbone):
    """Convolution causale longue paramétrée (Toeplitz).

    y[t] = Σ_{k=0}^{K-1} h[k] · x[t-k]

    Implémentée par FFT pour O(N log N) sur N long. K = kernel length, à
    régler ; par défaut K = max_seq_len pour expressivité max (la convolution
    reste paramétrée par K coefficients appris, pas N²).
    """

    def __init__(
        self, d_model: int, *, kernel_len: int, n_heads: int = 1, bias: bool = True
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.kernel_len = kernel_len
        self.n_heads = n_heads
        # Filtre par tête × dim : (n_heads, d_model, kernel_len) — chaque tête
        # apprend sa propre fonction de réponse. Init small pour stabilité.
        self.h = nn.Parameter(torch.randn(n_heads, d_model, kernel_len) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None
        # Mélange des têtes vers d_model : projection (n_heads * d_model -> d_model)
        if n_heads > 1:
            self.head_proj = nn.Linear(n_heads * d_model, d_model, bias=False)
        else:
            self.head_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, N, d_model) → (B, N, d_model)."""
        B, N, D = x.shape
        K = self.kernel_len
        # FFT-based causal conv. On pad x à droite, h à droite, puis FFT.
        fft_len = 1
        while fft_len < N + K - 1:
            fft_len *= 2

        # x_freq : (B, N, D) → (B, D, N) puis pad → (B, D, fft_len)
        x_perm = x.transpose(1, 2).contiguous()  # (B, D, N)
        # h : (H, D, K) → broadcast comme noyau commun à tous les batchs
        x_freq = torch.fft.rfft(x_perm, n=fft_len, dim=-1)         # (B, D, F)
        h_freq = torch.fft.rfft(self.h, n=fft_len, dim=-1)         # (H, D, F)

        # Pour chaque tête, output_head = ifft(x_freq * h_freq[head])
        # Vectorisé : (B, 1, D, F) × (1, H, D, F) → (B, H, D, F)
        prod = x_freq.unsqueeze(1) * h_freq.unsqueeze(0)            # (B, H, D, F)
        y_full = torch.fft.irfft(prod, n=fft_len, dim=-1)           # (B, H, D, fft_len)
        y = y_full[..., :N]                                          # (B, H, D, N) causal slice

        # Concat puis projection vers d_model
        y = y.transpose(2, 3).contiguous()                           # (B, H, N, D)
        if self.head_proj is None:
            y = y.squeeze(1)                                         # (B, N, D)
        else:
            y = y.permute(0, 2, 1, 3).contiguous().view(B, N, self.n_heads * D)
            y = self.head_proj(y)

        if self.bias is not None:
            y = y + self.bias
        return y

    @property
    def class_name(self) -> str:
        return "toeplitz"


class HankelSSMBackbone(Backbone):
    """SSM générique (Hankel-rank low) : h_{t+1} = A h_t + B x_t, y_t = C h_t.

    Paramétrage diagonal stable pour A (eigenvalues réelles dans [0, 1[ via
    sigmoid). Pas de HiPPO, pas de selectivity (DOC/03 §1.1 : on n'importe
    pas Mamba/S4).

    state_size = rang Hankel effectif issu de phase 2 (P-batterie Ho-Kalman).
    """

    def __init__(
        self, d_model: int, *, state_size: int, scan_chunk: int = 64
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.scan_chunk = scan_chunk
        # A diagonal (stable), B (d_model, state_size), C (state_size, d_model)
        # On paramètre A_log → A = sigmoid(A_log) ∈ (0, 1) pour stabilité.
        self.A_log = nn.Parameter(torch.randn(state_size) * 0.5 - 1.0)  # init ~ sigmoid(-1) = 0.27
        self.B = nn.Parameter(torch.randn(d_model, state_size) * 0.02)
        self.C = nn.Parameter(torch.randn(state_size, d_model) * 0.02)
        self.D = nn.Parameter(torch.zeros(d_model))  # skip connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, N, d_model) → (B, N, d_model).

        Récurrence linéaire dérivable. Pour efficacité on fait un scan
        Python (O(N) sur CPU/GPU). Pour très long N, utiliser une
        implémentation parallèle scan (CUDA), pas requis pour SMNIST.
        """
        B, N, D = x.shape
        A = torch.sigmoid(self.A_log)                                # (S,)
        # u = x @ B : (B, N, state_size)
        u = x @ self.B                                                # (B, N, S)
        # Récurrence : h_{t+1} = A * h_t + u_t
        h = torch.zeros(B, self.state_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(N):
            h = A.unsqueeze(0) * h + u[:, t, :]                       # (B, S)
            outputs.append(h)
        H = torch.stack(outputs, dim=1)                               # (B, N, S)
        # y = H @ C + D * x (skip)
        y = H @ self.C + self.D.view(1, 1, D) * x
        return y

    @property
    def class_name(self) -> str:
        return "hankel"


class CompositeBackbone(Backbone):
    """Somme additive de plusieurs backbones (composition phase 2 §A.3).

    Utile quand phase 2 identifie une combinaison Toeplitz + Hankel ou autre.
    """

    def __init__(self, components: list[Backbone]) -> None:
        super().__init__()
        if not components:
            raise ValueError("CompositeBackbone requiert au moins 1 composant")
        self.components = nn.ModuleList(components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.components[0](x)
        for comp in self.components[1:]:
            out = out + comp(x)
        return out

    @property
    def class_name(self) -> str:
        return "+".join(c.class_name for c in self.components)


def build_backbone_from_class(
    class_name: str, *, d_model: int, params: dict | None = None
) -> Backbone:
    """Instanciation Backbone à partir du nom de classe SCH (rapport phase 2).

    `params` : dict spécifique à la classe (ex. kernel_len pour Toeplitz,
    state_size pour Hankel). Si absent, valeurs par défaut sécurisées.
    """
    params = params or {}
    if class_name in ("toeplitz", "conv", "toeplitz_conv"):
        return ToeplitzConvBackbone(
            d_model=d_model,
            kernel_len=int(params.get("kernel_len", 256)),
            n_heads=int(params.get("n_heads", 1)),
        )
    if class_name in ("hankel", "ssm", "hankel_ssm"):
        return HankelSSMBackbone(
            d_model=d_model,
            state_size=int(params.get("state_size", 32)),
        )
    if class_name == "identity":
        from phase3_kernel_asp.backbone import IdentityBackbone
        return IdentityBackbone()
    if class_name == "linear":
        from phase3_kernel_asp.backbone import LinearBackbone
        return LinearBackbone(d_model=d_model)
    if "+" in class_name:
        # composite "toeplitz+hankel"
        sub_names = class_name.split("+")
        sub_params = params.get("components", [{} for _ in sub_names])
        components = [
            build_backbone_from_class(sn.strip(), d_model=d_model, params=sp)
            for sn, sp in zip(sub_names, sub_params, strict=False)
        ]
        return CompositeBackbone(components)
    raise ValueError(
        f"class_name inconnu : {class_name!r}. "
        f"Supportés : toeplitz, hankel, identity, linear, ou composite 'a+b'."
    )

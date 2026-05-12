"""
smart_init.py — extraction des vecteurs Smart Init Matriochka depuis les
dumps phase 2 (DOC/03 §5.2).

Schéma :
1. Recharger un dump d'attention (un fichier .pt phase 2 multi-bucket).
2. Pour chaque (layer, head, exemple) : SVD sur la matrice A → top-k vecteurs
   singuliers.
3. Trier les têtes par spec_h (variance r_eff sur régimes) — fourni par
   `phase2_audit_spectral.head_specialization`.
4. Garder les K_total premiers vecteurs (alignés colonnes U_max[:, :K_total]
   et V_max[:, :K_total]) pour les têtes les plus spécialisées.

Sortie : MatriochkaInitConfig prêt à passer à ASPTransformer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from phase3_kernel_asp.matriochka import MatriochkaInitConfig


def load_dumps(dump_dir: Path | str, glob: str = "audit_dump_seq*.pt") -> list[dict]:
    """Charge les dumps multi-bucket produits par run_extract.py V2."""
    files = sorted(Path(dump_dir).glob(glob))
    if not files:
        raise FileNotFoundError(f"Aucun dump matchant {glob} dans {dump_dir}")
    return [torch.load(str(f), map_location="cpu", weights_only=False) for f in files]


def compute_top_singular_vectors_per_head(
    dumps: list[dict],
    *,
    k_per_head: int = 4,
    max_examples_per_bucket: int = 16,
    seed: int = 0,
) -> dict[tuple[int, int], torch.Tensor]:
    """Pour chaque (layer, head), agrège top-k_per_head singular vectors moyens.

    Retourne {(layer, head) → (d, k_per_head)} où d = seq_len effectif.

    Note : les vecteurs U dans la SVD de A (matrice d'attention N×N) sont de
    dimension N (et non d_model). On retourne donc des vecteurs de dim N.
    Pour le smart init Matriochka qui veut des vecteurs de dim d_model, il
    faut faire un projection (linear) ou choisir un autre signal (cf. §5.2
    : "extraire vecteurs singuliers des têtes spécialisées").

    **Limite V1** : pour smart init Matriochka d'ASPLayer (d_model=256) à
    partir de dumps d'attention (dim N variable), on ne peut PAS injecter
    directement. Cette fonction sort les vecteurs en dim N — le caller doit
    décider de la projection.
    """
    if not dumps:
        return {}
    L = len(dumps[0]["attn"])
    H = dumps[0]["attn"][0].size(1)
    rng = np.random.default_rng(seed)

    out: dict[tuple[int, int], list[torch.Tensor]] = {}
    for dump in dumps:
        B = dump["attn"][0].size(0)
        idx = rng.choice(B, size=min(B, max_examples_per_bucket), replace=False)
        for ell in range(L):
            A_layer = dump["attn"][ell][idx]   # (B_sub, H, N, N)
            for h in range(H):
                A_head = A_layer[:, h]          # (B_sub, N, N)
                # SVD batchée
                try:
                    U, S, _Vt = torch.linalg.svd(A_head.float(), full_matrices=False)
                except Exception:
                    # Fallback eigvalsh sur AAt
                    AAt = A_head.float() @ A_head.float().transpose(-2, -1)
                    eye = torch.eye(AAt.size(-1), dtype=AAt.dtype, device=AAt.device)
                    AAt = AAt + 1e-6 * eye
                    eigvals, eigvecs = torch.linalg.eigh(AAt)
                    U = eigvecs.flip(-1)
                # U : (B_sub, N, N) — top-k singular vectors per example
                top_k_vecs = U[..., :k_per_head]  # (B_sub, N, k)
                # Moyenne sur exemples (vecteurs alignés par convention SVD)
                mean_vecs = top_k_vecs.mean(dim=0)  # (N, k)
                out.setdefault((ell, h), []).append(mean_vecs)

    # Pour chaque (ell, h), concat over buckets puis tri par norme (proxy d'importance)
    # Note : seq_len varie par bucket → on ne peut PAS simplement concat. On
    # fait moyenne au bucket le plus représenté ou pad/crop.
    result: dict[tuple[int, int], torch.Tensor] = {}
    for key, vecs_list in out.items():
        # On garde le bucket le plus grand seq_len (le plus riche structurellement)
        max_n = max(v.size(0) for v in vecs_list)
        chosen = next(v for v in vecs_list if v.size(0) == max_n)
        result[key] = chosen  # (N_max, k)
    return result


def make_smart_init_for_asp_layer(
    dump_dir: Path | str,
    *,
    d_model: int,
    R_max: int,
    K_per_layer: int = 16,
    k_per_head: int = 2,
    seed: int = 0,
) -> list[MatriochkaInitConfig]:
    """Produit une liste de MatriochkaInitConfig (1 par layer) pour
    ASPTransformer phase 3.

    **Adaptation V1** : les vecteurs singuliers sont de dim N (seq_len) alors
    que les bases U_max/V_max d'ASPLayer sont (d_model, R_max). On utilise
    donc une projection aléatoire fixée par seed (Johnson-Lindenstrauss) pour
    passer dim N → d_model. La structure relative entre vecteurs est
    largement préservée (théorème JL) ; les K_total premières colonnes
    U_max ont la même empreinte spectrale relative que les top heads
    Oracle.

    Si K_per_layer < R_max : les R_max - K_per_layer colonnes restantes
    sont initialisées en random (Xavier) côté ASPLayer.
    """
    dumps = load_dumps(dump_dir)
    vecs_per_head = compute_top_singular_vectors_per_head(
        dumps, k_per_head=k_per_head, seed=seed,
    )

    # Détermine L (couches) depuis les clés
    L = max(key[0] for key in vecs_per_head) + 1 if vecs_per_head else 6
    rng = torch.Generator().manual_seed(seed)

    configs: list[MatriochkaInitConfig] = []
    for ell in range(L):
        # Pour la couche ell : récolte tous les vecteurs des têtes
        layer_vecs: list[torch.Tensor] = []
        for (l_, h), v in vecs_per_head.items():
            if l_ == ell:
                layer_vecs.append(v)  # chacun (N, k_per_head)

        if not layer_vecs:
            configs.append(MatriochkaInitConfig(strategy="xavier"))
            continue

        # Concat → (total_vecs, N) avec total = H * k_per_head
        # Note : v est (N, k) → transpose en (k, N) puis concat → (sum_k, N)
        all_vecs = torch.cat([v.T for v in layer_vecs], dim=0)  # (H × k_per_head, N)

        # Garder les K_per_layer premiers (déjà ordonnés par head index — on
        # pourrait raffiner avec spec_h ranking mais V1 prend les premiers).
        K_total = min(K_per_layer, all_vecs.size(0), R_max)
        chosen = all_vecs[:K_total]  # (K_total, N)

        # Projection JL N → d_model (matrice gaussienne normalisée)
        N = chosen.size(1)
        jl_matrix = torch.randn(N, d_model, generator=rng) / (N ** 0.5)
        projected = chosen @ jl_matrix  # (K_total, d_model)
        # Re-normalise pour rester dans une échelle Xavier-friendly
        projected = projected / (projected.norm(dim=1, keepdim=True) + 1e-6)
        # MatriochkaInitConfig attend smart_init_vectors de shape (d, K_total)
        smart_vectors = projected.T  # (d_model, K_total)

        configs.append(MatriochkaInitConfig(
            strategy="smart",
            smart_init_vectors=smart_vectors,
            smart_init_freeze=True,
        ))
    return configs

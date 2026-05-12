"""
language.py — LLOracle : adapter pour Oracle de modélisation langagière (LL).

Spec : DOC/CATALOGUE §3.3 "TinyStories pour LL Sprint S7".

V2 complet : deux backends supportés
1. HFLanguageBackend : utilise transformers.AutoModelForCausalLM si dispo
2. MinimalLMBackend : utilise CODE/catalog/oracles/_minimal_transformer.py
   (GPT-style minimal) avec checkpoint torch.state_dict

Régime LL : omega = depth (profondeur structurelle des clauses imbriquées),
delta = seq_len fenêtre contexte. Le prompt template génère du texte avec
parenthèses imbriquées pour stresser la dépendance long-range.

NB : sans checkpoint pré-entraîné, les attentions seront aléatoires
(Xavier init). C'est utile pour :
- valider la pipeline end-to-end avant Sprint S7
- tester Properties sur attentions "génériques" comme baseline
- comparer Oracle entraîné vs Oracle aléatoire (test négatif Partie 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from catalog.oracles._minimal_transformer import (
    MinimalTransformer,
    MinimalTransformerSpec,
)
from catalog.oracles.base import AbstractOracle, AttentionDump, RegimeSpec


@dataclass
class LLModelSpec:
    vocab_size: int = 256  # char-level default
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 1024
    pad_id: int = 0
    eos_id: int = 1


# ----------------------------------------------------------------------------
# Backend abstraction
# ----------------------------------------------------------------------------

class _LMBackend:
    """Interface backend LM : tokenize + forward avec attentions."""

    n_layers: int
    n_heads: int

    def tokenize(self, texts: list[str], max_len: int) -> torch.Tensor:
        raise NotImplementedError

    def forward_with_attn(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError


class MinimalLMBackend(_LMBackend):
    """Backend basé sur MinimalTransformer (GPT-style) + tokenizer char-level."""

    def __init__(self, spec: LLModelSpec, checkpoint_path: str | Path | None = None,
                 device: str = "cpu") -> None:
        m_spec = MinimalTransformerSpec(
            vocab_size=spec.vocab_size, d_model=spec.d_model,
            n_heads=spec.n_heads, n_layers=spec.n_layers,
            d_ff=spec.d_ff, max_seq_len=spec.max_seq_len,
            causal=True, pad_id=spec.pad_id,
        )
        self.model = MinimalTransformer(m_spec)
        if checkpoint_path is not None:
            self.model.load_or_init(str(checkpoint_path))
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.n_layers = spec.n_layers
        self.n_heads = spec.n_heads
        self.spec = spec

    def tokenize(self, texts: list[str], max_len: int) -> torch.Tensor:
        """Tokenizer char-level : chaque caractère est un token (0-255)."""
        out = torch.full((len(texts), max_len), self.spec.pad_id, dtype=torch.long)
        for i, txt in enumerate(texts):
            ids = [min(ord(c), self.spec.vocab_size - 1) for c in txt[:max_len]]
            out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    @torch.no_grad()
    def forward_with_attn(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        input_ids = input_ids.to(self.device)
        out = self.model(input_ids, output_attentions=True)
        return out["attentions"]  # list of (B, H, N, N)


class HFLanguageBackend(_LMBackend):
    """Backend HuggingFace AutoModelForCausalLM. Requiert `transformers`.

    Download HuggingFace Hub avec retry automatique (3 tentatives, backoff
    exponentiel) — utile sur pod où la connexion peut être instable.

    Usage :
        backend = HFLanguageBackend("meta-llama/Llama-3.2-1B", device="cuda")
        oracle = LLOracle(backend=backend, ...)
    """

    def __init__(self, model_name_or_path: str, device: str = "cpu",
                 torch_dtype: torch.dtype = torch.float32,
                 max_attempts: int = 3) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "HFLanguageBackend requiert `transformers`. "
                "uv add transformers ou utiliser MinimalLMBackend."
            ) from e
        from shared.retry import retry_call
        import logging
        log = logging.getLogger("catalog.oracles.language")

        # Retry sur les téléchargements Hub (réseau pod instable)
        self.tokenizer = retry_call(
            AutoTokenizer.from_pretrained, args=(model_name_or_path,),
            max_attempts=max_attempts, base_delay=2.0, jitter=0.5,
            logger=log,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = retry_call(
            AutoModelForCausalLM.from_pretrained,
            args=(model_name_or_path,),
            kwargs={"torch_dtype": torch_dtype, "attn_implementation": "eager"},
            max_attempts=max_attempts, base_delay=2.0, jitter=0.5,
            logger=log,
        )
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.n_layers = self.model.config.num_hidden_layers
        self.n_heads = self.model.config.num_attention_heads
        log.info("HFLanguageBackend loaded: %s (n_layers=%d, n_heads=%d)",
                 model_name_or_path, self.n_layers, self.n_heads)

    def tokenize(self, texts: list[str], max_len: int) -> torch.Tensor:
        enc = self.tokenizer(texts, padding="max_length", truncation=True,
                             max_length=max_len, return_tensors="pt")
        return enc["input_ids"]

    @torch.no_grad()
    def forward_with_attn(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        input_ids = input_ids.to(self.device)
        out = self.model(input_ids, output_attentions=True, use_cache=False)
        return list(out.attentions)  # tuple of (B, H, N, N)


# ----------------------------------------------------------------------------
# Prompt templates
# ----------------------------------------------------------------------------

def nested_parentheses_template(depth: int, seq_len: int, seed: int = 0) -> str:
    """Génère du texte avec parenthèses imbriquées au depth demandé.

    'The cat (that the dog (that the mouse saw) chased) ran.'
    Plus depth est grand, plus la dépendance long-range est forte.

    Le texte est ensuite paddé avec mots remplissage jusqu'à seq_len chars.
    """
    rng = torch.Generator().manual_seed(seed)
    subjects = ["The cat", "The dog", "A man", "A woman", "The child"]
    verbs = ["saw", "chased", "knew", "ran"]
    obj = ["yesterday", "the book", "a friend", "the park"]

    def _nested(d: int) -> str:
        if d == 0:
            i = int(torch.randint(0, len(verbs), (1,), generator=rng).item())
            j = int(torch.randint(0, len(obj), (1,), generator=rng).item())
            return f"{verbs[i]} {obj[j]}"
        si = int(torch.randint(0, len(subjects), (1,), generator=rng).item())
        return f"{subjects[si]} (that {_nested(d-1)})"

    body = _nested(depth) + " ran away."
    # Pad avec ponctuation
    pad_text = " The story continues with details and more details and more details."
    while len(body) < seq_len:
        body += pad_text
    return body[:seq_len]


def random_prompt_template(complexity: int, seq_len: int, seed: int = 0) -> str:
    """Fallback : texte pseudo-aléatoire avec entropie variable.

    complexity = 0 : caractères majoritairement 'a' (basse entropie)
    complexity élevé : caractères uniformes 32-126 (haute entropie)
    """
    rng = torch.Generator().manual_seed(seed)
    n_distinct = max(1, complexity * 4)
    # Vocab : 32..32+n_distinct (caractères imprimables)
    vocab_max = 32 + min(n_distinct, 95)
    ids = torch.randint(32, vocab_max, (seq_len,), generator=rng).tolist()
    return "".join(chr(i) for i in ids)


# ----------------------------------------------------------------------------
# LLOracle
# ----------------------------------------------------------------------------

class LLOracle(AbstractOracle):
    """Oracle LM (TinyStories, Llama-3.2-1B, BabyLM, etc.).

    Args
    ----
    backend : _LMBackend (MinimalLMBackend ou HFLanguageBackend)
    prompt_fn : callable (depth, seq_len, seed) → str. Défaut = nested_parentheses.
    n_examples_per_regime_max : cap pour éviter OOM
    oracle_id : identifiant unique pour MLflow
    """

    domain = "ll"

    def __init__(
        self,
        *,
        backend: _LMBackend | None = None,
        checkpoint_path: str | Path | None = None,
        model_spec: LLModelSpec | None = None,
        prompt_fn: Callable[[int, int, int], str] = nested_parentheses_template,
        device: str = "cpu",
        oracle_id: str | None = None,
        n_examples_max: int = 64,
    ) -> None:
        # Backward compat : si pas de backend explicite, on construit un
        # MinimalLMBackend avec ou sans checkpoint
        if backend is None:
            if checkpoint_path is not None:
                ckpt = Path(checkpoint_path)
                if not ckpt.is_file():
                    raise FileNotFoundError(f"Checkpoint LL introuvable : {ckpt}")
            spec = model_spec or LLModelSpec()
            backend = MinimalLMBackend(spec, checkpoint_path=checkpoint_path, device=device)
            stem = Path(checkpoint_path).stem if checkpoint_path else "random_init"
            self.oracle_id = oracle_id or f"ll_minimal_{stem}"
        else:
            self.oracle_id = oracle_id or "ll_external"
        self.backend = backend
        self.prompt_fn = prompt_fn
        self.device = device
        self.n_layers = backend.n_layers
        self.n_heads = backend.n_heads
        self.n_examples_max = n_examples_max

    def extract_regime(
        self, regime: RegimeSpec, n_examples: int
    ) -> AttentionDump:
        depth = int(regime.omega) if regime.omega is not None else 1
        seq_len = int(regime.delta) if regime.delta is not None else 64
        # Cap par le max_seq_len du backend pour éviter erreurs out-of-range
        max_supported = getattr(getattr(self.backend, "spec", None), "max_seq_len", 1024)
        seq_len = min(seq_len, max_supported, 1024)
        n_examples = min(n_examples, self.n_examples_max)

        texts = [self.prompt_fn(depth, seq_len, seed=i) for i in range(n_examples)]
        input_ids = self.backend.tokenize(texts, max_len=seq_len)
        attns = self.backend.forward_with_attn(input_ids)
        # Cast en FP64 si MinimalLM ; HF peut être fp16
        attns_fp64 = [a.to(torch.float64).cpu() for a in attns]

        return AttentionDump(
            attn=attns_fp64,
            omegas=torch.full((n_examples,), float(depth)),
            deltas=torch.full((n_examples,), float(seq_len)),
            entropies=torch.zeros(n_examples),
            tokens=input_ids.cpu(),
            query_pos=torch.zeros(n_examples, dtype=torch.long),  # pas de classification
            metadata={
                "oracle_id": self.oracle_id,
                "depth": depth,
                "seq_len_target": seq_len,
                "domain": self.domain,
            },
        )

    def regime_grid(self) -> list[RegimeSpec]:
        depths = [0, 1, 2, 4, 8]
        seq_lens = [64, 256, 1024]
        out: list[RegimeSpec] = []
        for d in depths:
            for s in seq_lens:
                out.append(RegimeSpec(omega=d, delta=s, entropy=0.0))
        return out


def make_tinystories_template() -> Callable[[int, int, int], str]:
    """Factory : template prompt orienté TinyStories (nested clauses)."""
    return nested_parentheses_template

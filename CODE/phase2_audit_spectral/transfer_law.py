"""
transfer_law.py — loi de transfert r_target = f(ω, Δ, ℋ).

Spec : DOC/02 §2.4. Forme multiplicative testée :
    r_target ≈ a · ω^α · Δ^β · g(ℋ)

Dans la version V1, on fit en log-linéaire :
    log r_target = log a + α log(1+ω) + β log(1+Δ) + γ_ent · ℋ

(le +1 évite log(0) sur les régimes ω=0 ou Δ=0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TransferLawFit:
    log_a: float
    alpha: float       # exposant ω
    beta: float        # exposant Δ
    gamma: float       # coefficient ℋ
    r2: float
    residuals: np.ndarray
    n: int

    def predict(self, omega: np.ndarray, delta: np.ndarray, entropy: np.ndarray) -> np.ndarray:
        return np.exp(
            self.log_a
            + self.alpha * np.log1p(omega)
            + self.beta * np.log1p(delta)
            + self.gamma * entropy
        )


def fit_transfer_law(
    *,
    r_target: np.ndarray,
    omega: np.ndarray,
    delta: np.ndarray,
    entropy: np.ndarray,
) -> TransferLawFit:
    """Fit log-linéaire de r_target sur (ω, Δ, ℋ)."""
    # Filtre les r_target non strictement positifs
    mask = r_target > 0
    r = r_target[mask].astype(np.float64)
    o = omega[mask].astype(np.float64)
    d = delta[mask].astype(np.float64)
    h = entropy[mask].astype(np.float64)
    if r.size < 4:
        raise ValueError(f"Trop peu d'observations valides ({r.size}) pour fit.")

    y = np.log(r)
    X = np.stack([np.ones_like(o), np.log1p(o), np.log1p(d), h], axis=1)
    coef, residuals_sum, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    log_a, alpha, beta, gamma = coef
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return TransferLawFit(
        log_a=float(log_a), alpha=float(alpha), beta=float(beta), gamma=float(gamma),
        r2=r2, residuals=y - pred, n=r.size,
    )


def cross_domain_compare(fits: dict[str, TransferLawFit]) -> dict[str, float | str]:
    """Compare les exposants entre domaines (DOC/02 §2.4 : SCH universelle ?
    domain-spécifique ? partiellement universelle ?).
    """
    if len(fits) < 2:
        return {"verdict": "n/a", "n_domains": len(fits)}
    alphas = np.array([f.alpha for f in fits.values()])
    betas = np.array([f.beta for f in fits.values()])
    gammas = np.array([f.gamma for f in fits.values()])
    # Critère heuristique : variance des exposants relative à leur moyenne
    def _cv(x: np.ndarray) -> float:
        m = abs(x.mean())
        return float(x.std() / m) if m > 1e-6 else float("inf")

    cv_alpha = _cv(alphas)
    cv_beta = _cv(betas)
    cv_gamma = _cv(gammas)
    universal_threshold = 0.15
    if max(cv_alpha, cv_beta, cv_gamma) < universal_threshold:
        verdict = "universal"
    elif min(cv_alpha, cv_beta, cv_gamma) < universal_threshold:
        verdict = "partially_universal"
    else:
        verdict = "domain_specific"
    return {
        "verdict": verdict,
        "cv_alpha": cv_alpha,
        "cv_beta": cv_beta,
        "cv_gamma": cv_gamma,
        "n_domains": len(fits),
    }

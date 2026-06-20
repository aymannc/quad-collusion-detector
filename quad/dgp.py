"""Synthetic data-generating processes for the collusion-detection experiments.

Each regime returns (X, Y1, Y2) where X is (n, x_dim) shared input and Y1, Y2
are the two agents' outputs:

    independent          - agents react to shared X with independent noise
                           (conditionally independent given X)
    colluding            - agents share a hidden LINEAR signal Z beyond X
                           (conditionally dependent given X)
    honest_competent     - independent, but high competence (low noise) -- the
                           critical false-positive stress test
    nonlinear_collusion  - agents share a hidden signal that couples them
                           NONLINEARLY, so they are *linearly* uncorrelated
                           given X yet conditionally dependent. Linear partial
                           correlation FAILS here; CMI / KCI / residual-dCor
                           succeed. (Added to justify CMI over cheap linear
                           baselines on the reviewers' request.)

N (`n`) and the conditioning dimension (`x_dim`) are configurable so the
experiments can be scaled up and X enriched.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "independent", "colluding", "honest_competent", "nonlinear_collusion",
    "REGIMES",
]


def _shared_signal(rng, n, x_dim):
    """Shared input X (n, x_dim) and the scalar signal both agents track.
    At x_dim=1 the signal equals X, reproducing the original experiments."""
    X = rng.standard_normal((n, x_dim))
    signal = X.sum(axis=1) / np.sqrt(x_dim)  # unit variance
    return X, signal


def independent(seed, n=500, x_dim=1, noise=0.5):
    rng = np.random.default_rng(seed)
    X, s = _shared_signal(rng, n, x_dim)
    return X, s + rng.standard_normal(n) * noise, s + rng.standard_normal(n) * noise


def colluding(seed, signal_strength=0.4, n=500, x_dim=1, noise=0.5):
    rng = np.random.default_rng(seed)
    X, s = _shared_signal(rng, n, x_dim)
    Z = rng.standard_normal(n)  # hidden coordination signal beyond X
    y1 = s + signal_strength * Z + rng.standard_normal(n) * noise
    y2 = s + signal_strength * Z + rng.standard_normal(n) * noise
    return X, y1, y2


def honest_competent(seed, competence=0.95, n=500, x_dim=1):
    rng = np.random.default_rng(seed)
    X, s = _shared_signal(rng, n, x_dim)
    sigma = 0.01 if competence >= 0.999 else np.sqrt(1.0 / competence ** 2 - 1.0)
    return X, s + rng.standard_normal(n) * sigma, s + rng.standard_normal(n) * sigma


def nonlinear_collusion(seed, signal_strength=1.0, n=500, x_dim=1, noise=0.3):
    rng = np.random.default_rng(seed)
    X, s = _shared_signal(rng, n, x_dim)
    U = rng.standard_normal(n)  # hidden shared driver
    y1 = s + U + rng.standard_normal(n) * noise
    # corr(U, U**2 - 1) = 0 -> linearly uncorrelated, but clearly dependent
    y2 = s + signal_strength * (U ** 2 - 1.0) + rng.standard_normal(n) * noise
    return X, y1, y2


REGIMES = {
    "independent": independent,
    "colluding": colluding,
    "honest_competent": honest_competent,
    "nonlinear_collusion": nonlinear_collusion,
}

"""Baseline dependence / conditional-independence tests.

These are the comparators the reviewers asked for, so that CMI is benchmarked
against more than the (deliberately weak) unconditional Pearson test:

    pearson_test          - unconditional linear correlation (straw-man)
    partial_corr_test     - linear *conditional* test (regress out X)
    distance_corr         - nonlinear unconditional dependence (statistic only)
    residual_dcor_test    - nonlinear *conditional* proxy (dCor of X-residuals)
    hsic_test             - kernel dependence (unconditional), permutation null
    kci_test              - kernel conditional independence  [needs causal-learn]

Every *_test returns (statistic, p_value).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, t as _t
from scipy.spatial.distance import cdist, pdist, squareform

__all__ = [
    "pearson_test", "partial_corr_test", "distance_corr",
    "residual_dcor_test", "hsic_test", "kci_test",
]


def _as2d(a):
    a = np.asarray(a, float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def pearson_test(Y1, Y2, n_perm: int = 200, rng=None):
    """Unconditional |Pearson r| with a permutation null (matches the paper's
    Table III baseline)."""
    rng = np.random.default_rng() if rng is None else rng
    Y1 = np.asarray(Y1, float); Y2 = np.asarray(Y2, float)
    r_obs = abs(pearsonr(Y1, Y2)[0])
    null = np.array([abs(pearsonr(Y1, rng.permutation(Y2))[0]) for _ in range(n_perm)])
    return r_obs, (1 + int((null >= r_obs).sum())) / (n_perm + 1)


def _residualize(Y, X):
    X = _as2d(X)
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    return Y - Xd @ beta


def partial_corr_test(Y1, Y2, X):
    """Linear partial correlation of (Y1, Y2) given X, with an analytic t-test."""
    X = _as2d(X)
    r1, r2 = _residualize(np.asarray(Y1, float), X), _residualize(np.asarray(Y2, float), X)
    r = pearsonr(r1, r2)[0]
    dof = len(r1) - 2 - X.shape[1]
    if dof <= 0 or abs(r) >= 1.0:
        return abs(float(r)), 1.0
    tval = r * np.sqrt(dof / (1.0 - r ** 2))
    return abs(float(r)), float(2 * _t.sf(abs(tval), dof))


def distance_corr(a, b) -> float:
    """Szekely distance correlation (nonlinear, unconditional)."""
    a, b = _as2d(a), _as2d(b)
    A = cdist(a, a); B = cdist(b, b)
    A = A - A.mean(0) - A.mean(1)[:, None] + A.mean()
    B = B - B.mean(0) - B.mean(1)[:, None] + B.mean()
    dcov = np.sqrt(max((A * B).mean(), 0.0))
    va = np.sqrt(max((A * A).mean(), 0.0)); vb = np.sqrt(max((B * B).mean(), 0.0))
    return float(dcov / np.sqrt(va * vb)) if va * vb > 0 else 0.0


def residual_dcor_test(Y1, Y2, X, n_perm: int = 200, rng=None):
    """Nonlinear conditional proxy: distance correlation of the X-residuals,
    with a permutation null. Catches nonlinear collusion that linear partial
    correlation misses."""
    rng = np.random.default_rng() if rng is None else rng
    r1 = _residualize(np.asarray(Y1, float), X)
    r2 = _residualize(np.asarray(Y2, float), X)
    obs = distance_corr(r1, r2)
    null = np.array([distance_corr(r1, rng.permutation(r2)) for _ in range(n_perm)])
    return obs, (1 + int((null >= obs).sum())) / (n_perm + 1)


def _rbf_kernel(M):
    M = _as2d(M)
    D = squareform(pdist(M, "sqeuclidean"))
    med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
    return np.exp(-D / (med + 1e-12))


def hsic_test(Y1, Y2, n_perm: int = 200, rng=None):
    """Hilbert-Schmidt Independence Criterion (kernel dependence), with a
    permutation null. Unconditional; use as a dependence-measure comparator."""
    rng = np.random.default_rng() if rng is None else rng
    a, b = _as2d(Y1), _as2d(Y2)
    n = len(a)
    H = np.eye(n) - 1.0 / n
    Kc = H @ _rbf_kernel(a) @ H

    def stat(Lmat):
        return float(np.sum(Kc * (H @ Lmat @ H)) / (n * n))

    obs = stat(_rbf_kernel(b))
    null = np.array([stat(_rbf_kernel(b[rng.permutation(n)])) for _ in range(n_perm)])
    return obs, (1 + int((null >= obs).sum())) / (n_perm + 1)


def kci_test(Y1, Y2, X):
    """Kernel Conditional Independence test (Zhang et al., 2011).

    Requires the optional dependency `causal-learn`. Returns (None, p_value).
    """
    try:
        from causallearn.utils.cit import CIT
    except Exception as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "kci_test requires 'causal-learn' (pip install causal-learn)"
        ) from exc
    data = np.column_stack([_as2d(Y1), _as2d(Y2), _as2d(X)])
    obj = CIT(data, "kci")
    z = list(range(2, data.shape[1]))
    return None, float(obj(0, 1, z))

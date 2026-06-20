"""Conditional mutual information estimator and significance test.

I(Y1; Y2 | X) via the Frenzel-Pompe k-nearest-neighbour estimator
(Chebyshev / max-norm), with a *local-conditional* permutation test for
significance: Y2 is reshuffled only among each point's nearest neighbours in
X-space, which preserves p(Y2 | X) under the conditional-independence null.

Implementation note
-------------------
Neighbour counting uses a KD-tree (scipy.spatial.cKDTree), which is
O(N log N) in time and O(N) in memory. This replaces the earlier O(N^2)
pairwise-distance-matrix version so the estimator scales to large N
(tens of thousands of samples) without exhausting memory. The numerical
estimator (Frenzel-Pompe) is unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma
from scipy.spatial import cKDTree

__all__ = ["cmi_estimate", "cmi_permutation_test"]

_INF = np.inf


def _as2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def cmi_estimate(Y1, Y2, X, k: int = 5) -> float:
    """Frenzel-Pompe estimate of I(Y1; Y2 | X).

    Y1, Y2 are 1-D arrays of length n; X is (n,) or (n, d).
    """
    Y1, Y2, X = _as2d(Y1), _as2d(Y2), _as2d(X)
    n = len(Y1)
    k = max(1, min(k, n - 1))

    joint = np.concatenate([X, Y1, Y2], axis=1)
    XY1 = np.concatenate([X, Y1], axis=1)
    XY2 = np.concatenate([X, Y2], axis=1)

    # distance to the k-th neighbour in joint space (Chebyshev), excluding self
    d_joint, _ = cKDTree(joint).query(joint, k=k + 1, p=_INF)
    eps = np.nextafter(d_joint[:, k], 0.0)      # strict-less radius (open ball)
    eps = np.maximum(eps, 1e-12)

    def count_within(points):
        tree = cKDTree(points)
        c = tree.query_ball_point(points, eps, p=_INF, return_length=True)
        return np.asarray(c) - 1                # drop self

    n_x = count_within(X)
    n_xy1 = count_within(XY1)
    n_xy2 = count_within(XY2)
    return float(
        digamma(k)
        + np.mean(digamma(n_x + 1) - digamma(n_xy1 + 1) - digamma(n_xy2 + 1))
    )


def cmi_permutation_test(
    Y1, Y2, X, k: int = 5, n_perm: int = 200, n_local: int = 10, rng=None
):
    """Local-conditional permutation p-value for I(Y1; Y2 | X).

    Returns (cmi_observed, p_value, null_distribution).
    """
    rng = np.random.default_rng() if rng is None else rng
    Y1, Y2, X = _as2d(Y1), _as2d(Y2), _as2d(X)
    n = len(Y1)

    obs = cmi_estimate(Y1, Y2, X, k=k)

    # each point's nearest neighbours in X-space (excluding self)
    L = max(1, min(n_local, n - 1))
    _, nbrs = cKDTree(X).query(X, k=L + 1, p=_INF)
    nbrs = nbrs[:, 1:]                          # drop self column

    y2 = Y2[:, 0]
    null = np.empty(n_perm)
    for p in range(n_perm):
        cols = rng.integers(0, L, size=n)
        perm = nbrs[np.arange(n), cols]
        null[p] = cmi_estimate(Y1, y2[perm], X, k=k)

    p_value = (1 + int((null >= obs).sum())) / (n_perm + 1)
    return obs, float(p_value), null

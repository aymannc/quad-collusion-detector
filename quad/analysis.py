"""Run the CMI detector and the baseline comparators on (Y1, Y2, X) and
assemble result rows/tables."""
from __future__ import annotations

import numpy as np

from .cmi import cmi_permutation_test
from . import baselines as B


def run_all(Y1, Y2, X, k=5, n_perm=200, n_local=10, rng=None,
            use_hsic=False, use_kci=False):
    """Return {method: (statistic, p_value)} for CMI + baselines."""
    rng = np.random.default_rng() if rng is None else rng
    out = {}
    cmi, p, _ = cmi_permutation_test(Y1, Y2, X, k=k, n_perm=n_perm,
                                     n_local=n_local, rng=rng)
    out["CMI"] = (cmi, p)
    out["Pearson"] = B.pearson_test(Y1, Y2, n_perm=n_perm, rng=rng)
    out["PartialCorr"] = B.partial_corr_test(Y1, Y2, X)
    out["ResidualDCor"] = B.residual_dcor_test(Y1, Y2, X, n_perm=n_perm, rng=rng)
    if use_hsic:
        out["HSIC"] = B.hsic_test(Y1, Y2, n_perm=n_perm, rng=rng)
    if use_kci:
        try:
            out["KCI"] = B.kci_test(Y1, Y2, X)
        except ImportError:
            pass
    return out


def fires(result_row, alpha=0.05):
    """Boolean fire/no-fire per method at significance alpha."""
    return {m: (p < alpha) for m, (_, p) in result_row.items()}

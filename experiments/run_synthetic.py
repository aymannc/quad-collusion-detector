"""Synthetic experiments for the QUAD collusion detector.

Reproduces the paper's experiments and adds (1) stronger baselines and (2) a
nonlinear-collusion regime that justifies CMI over cheap linear tests.

  R1   false-positive control (independent agents)
  R2   detection-power sweep across collusion signal strengths
  R3   collusion-vs-competence separability sweep
  T3   head-to-head: all methods x {independent, colluding, honest, nonlinear}

Scale parameters are CLI flags so N (`--n`) and the conditioning dimension
(`--x-dim`) can be increased. Progress is logged to the console and to
results/run_<timestamp>.log with elapsed time and ETA. Use --quick for a fast
smoke run.

Examples
--------
    python -m experiments.run_synthetic --quick
    python -m experiments.run_synthetic --seeds 200 --n 500
    python -m experiments.run_synthetic --table3-only --seeds 200 --use-hsic
    tail -f results/run_*.log        # watch a long run from another terminal
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time

import numpy as np

from quad.cmi import cmi_permutation_test
from quad import baselines as B
from quad import dgp
from quad.runlog import setup_logging, Progress, fmt_dur

log = logging.getLogger("quad")


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def _fire_rate(gen, gen_kwargs, seeds, base_seed, cfg, label):
    prog = Progress(log, seeds, label, cfg.log_every)
    fired = 0
    for s in range(seeds):
        rng = np.random.default_rng(base_seed + s)
        X, Y1, Y2 = gen(base_seed + s, n=cfg.n, **gen_kwargs)
        _, p, _ = cmi_permutation_test(Y1, Y2, X, k=cfg.k, n_perm=cfg.n_perm,
                                       n_local=cfg.n_local, rng=rng)
        fired += int(p < cfg.alpha)
        prog.update(1, extra=f"running fire-rate={fired/(s+1):.3f}")
    return fired


def run_r1(cfg):
    log.info("[R1] false-positive control — %d seeds, n=%d, x_dim=%d",
             cfg.seeds, cfg.n, cfg.x_dim)
    f = _fire_rate(dgp.independent, {"x_dim": cfg.x_dim}, cfg.seeds, 0, cfg, "R1")
    lo, hi = _wilson(f, cfg.seeds)
    log.info("[R1] RESULT: %d/%d = %.4f  Wilson95 [%.4f, %.4f]  (target alpha=%.3f)",
             f, cfg.seeds, f / cfg.seeds, lo, hi, cfg.alpha)
    return [{"experiment": "R1", "fire_rate": f / cfg.seeds, "ci_lo": lo, "ci_hi": hi}]


def run_r2(cfg):
    signals = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)
    if cfg.quick:
        signals = (0.0, 0.20, 0.40, 0.60)
    log.info("[R2] detection power — %d signals x %d seeds", len(signals), cfg.seeds)
    rows = []
    for i, sig in enumerate(signals):
        f = _fire_rate(dgp.colluding, {"x_dim": cfg.x_dim, "signal_strength": sig},
                       cfg.seeds, 10000 * (i + 1), cfg, f"R2 signal={sig:.2f}")
        lo, hi = _wilson(f, cfg.seeds)
        rows.append({"signal": sig, "power": f / cfg.seeds, "ci_lo": lo, "ci_hi": hi})
        log.info("[R2] signal=%.2f -> power=%.3f", sig, f / cfg.seeds)
    return rows


def run_r3(cfg):
    comps = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98)
    if cfg.quick:
        comps = (0.70, 0.90, 0.98)
    log.info("[R3] separability — %d competences x %d seeds", len(comps), cfg.seeds)
    rows = []
    for i, c in enumerate(comps):
        f = _fire_rate(dgp.honest_competent, {"x_dim": cfg.x_dim, "competence": c},
                       cfg.seeds, 20000 * (i + 1), cfg, f"R3 competence={c:.2f}")
        lo, hi = _wilson(f, cfg.seeds)
        rows.append({"competence": c, "fp_rate": f / cfg.seeds, "ci_lo": lo, "ci_hi": hi})
        log.info("[R3] competence=%.2f -> fp_rate=%.4f", c, f / cfg.seeds)
    return rows


def run_table3(cfg):
    """Head-to-head fire rates for every method across four regimes."""
    regimes = [
        ("independent", "NOT fire", dgp.independent, {"x_dim": cfg.x_dim}),
        ("colluding(0.4)", "fire", dgp.colluding, {"x_dim": cfg.x_dim, "signal_strength": 0.4}),
        ("honest(0.95)", "NOT fire", dgp.honest_competent, {"x_dim": cfg.x_dim, "competence": 0.95}),
        ("nonlinear", "fire", dgp.nonlinear_collusion, {"x_dim": cfg.x_dim, "signal_strength": 1.0}),
    ]
    methods = ["CMI", "Pearson", "PartialCorr", "ResidualDCor"] + (["HSIC"] if cfg.use_hsic else [])
    log.info("[T3] head-to-head — %d seeds x %d regimes; methods=%s",
             cfg.seeds, len(regimes), methods)
    rows = []
    for ri, (name, exp, gen, kw) in enumerate(regimes):
        counts = {m: 0 for m in methods}
        prog = Progress(log, cfg.seeds, f"T3 {name}", cfg.log_every)
        for s in range(cfg.seeds):
            rng = np.random.default_rng(50000 * (ri + 1) + s)
            X, Y1, Y2 = gen(50000 * (ri + 1) + s, n=cfg.n, **kw)
            counts["CMI"] += cmi_permutation_test(Y1, Y2, X, k=cfg.k, n_perm=cfg.n_perm, n_local=cfg.n_local, rng=rng)[1] < cfg.alpha
            counts["Pearson"] += B.pearson_test(Y1, Y2, n_perm=cfg.n_perm, rng=rng)[1] < cfg.alpha
            counts["PartialCorr"] += B.partial_corr_test(Y1, Y2, X)[1] < cfg.alpha
            counts["ResidualDCor"] += B.residual_dcor_test(Y1, Y2, X, n_perm=cfg.n_perm, rng=rng)[1] < cfg.alpha
            if cfg.use_hsic:
                counts["HSIC"] += B.hsic_test(Y1, Y2, n_perm=cfg.n_perm, rng=rng)[1] < cfg.alpha
            prog.update(1)
        row = {"regime": name, "expected": exp}
        for m in methods:
            row[m] = counts[m] / cfg.seeds
        rows.append(row)
        log.info("[T3] %-16s %-9s %s", name, exp,
                 "  ".join(f"{m}={row[m]:.3f}" for m in methods))
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--n", type=int, default=500, help="samples per run (N)")
    ap.add_argument("--x-dim", type=int, default=1, help="conditioning dimension")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-local", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quick", action="store_true", help="fast smoke config")
    ap.add_argument("--table3-only", action="store_true")
    ap.add_argument("--use-hsic", action="store_true")
    ap.add_argument("--log-every", type=int, default=25, help="progress log interval (seeds)")
    ap.add_argument("--logfile", default=None, help="explicit log file path")
    cfg = ap.parse_args()
    if cfg.quick:
        cfg.seeds = min(cfg.seeds, 20)
        cfg.n = min(cfg.n, 150)
        cfg.n_perm = min(cfg.n_perm, 60)
        cfg.log_every = min(cfg.log_every, 5)
    os.makedirs(cfg.outdir, exist_ok=True)

    setup_logging(cfg.outdir, logfile=cfg.logfile)
    log.info("QUAD synthetic experiments | config: seeds=%d n=%d x_dim=%d n_perm=%d k=%d alpha=%.3f%s",
             cfg.seeds, cfg.n, cfg.x_dim, cfg.n_perm, cfg.k, cfg.alpha,
             " [QUICK]" if cfg.quick else "")
    t0 = time.time()
    if cfg.table3_only:
        _write_csv(os.path.join(cfg.outdir, "table3_methods.csv"), run_table3(cfg))
    else:
        _write_csv(os.path.join(cfg.outdir, "r1_fp.csv"), run_r1(cfg))
        _write_csv(os.path.join(cfg.outdir, "r2_power.csv"), run_r2(cfg))
        _write_csv(os.path.join(cfg.outdir, "r3_separability.csv"), run_r3(cfg))
        _write_csv(os.path.join(cfg.outdir, "table3_methods.csv"), run_table3(cfg))
    log.info("ALL DONE in %s. Results in %s/", fmt_dur(time.time() - t0), cfg.outdir)


if __name__ == "__main__":
    main()

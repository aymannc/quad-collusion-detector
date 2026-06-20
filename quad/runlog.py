"""Lightweight logging + progress helpers for the experiment scripts.

Logs to both the console and a timestamped file under the output directory, so
long runs can be tailed (``tail -f results/run_*.log``) and audited later.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime

__all__ = ["setup_logging", "Progress", "fmt_dur"]


def setup_logging(outdir="results", name="quad", logfile=None, level=logging.INFO):
    """Configure a logger that writes to stdout and to outdir/run_<ts>.log."""
    os.makedirs(outdir, exist_ok=True)
    if logfile is None:
        logfile = os.path.join(outdir, f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info("log file: %s", logfile)
    return logger


def fmt_dur(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Progress:
    """Logs '<label>: i/total [elapsed, ETA]' every `every` updates."""

    def __init__(self, logger, total, label, every=25):
        self.log = logger
        self.total = max(1, total)
        self.label = label
        self.every = max(1, every)
        self.t0 = time.time()
        self.done = 0

    def update(self, n=1, extra=""):
        self.done += n
        if self.done % self.every == 0 or self.done >= self.total:
            elapsed = time.time() - self.t0
            rate = self.done / elapsed if elapsed > 0 else 0.0
            eta = (self.total - self.done) / rate if rate > 0 else 0.0
            msg = (f"  {self.label}: {self.done}/{self.total} "
                   f"[{fmt_dur(elapsed)} elapsed, ETA {fmt_dur(eta)}]")
            if extra:
                msg += f" {extra}"
            self.log.info(msg)

    @property
    def elapsed(self):
        return time.time() - self.t0

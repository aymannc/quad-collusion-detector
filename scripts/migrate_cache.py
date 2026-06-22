#!/usr/bin/env python3
"""One-off: upgrade old-format response caches to the current format.

Early runs saved responses as `responses_<a>-<b>_n<N>_s<seed>.json` with no
dataset segment and no content fingerprint. The current pipeline expects a
dataset-tagged filename and a `fingerprint` field (so caches can never be reused
across datasets). Files without a dataset tag predate dataset support and are
therefore the SYNTHETIC dataset.

This script rewrites each such file to
`responses_<a>-<b>_synthetic_n<N>_s<seed>.json` with the correct fingerprint, so
the finished synthetic runs are reused (not re-collected / re-charged). It only
migrates a file when the fingerprint of its stored data matches what the current
code regenerates for that seed — guaranteeing the responses still correspond to
the data, otherwise it is left untouched.

Usage:  python scripts/migrate_cache.py [--apply]   (default: dry-run)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quad import scenarios as S  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
# old format: no dataset segment between providers and nN
OLD = re.compile(r"^responses_([a-z0-9]+-[a-z0-9]+)_n(\d+)_s(\d+)\.json$")


def main(apply: bool) -> None:
    migrated = skipped = 0
    for path in sorted(glob.glob(os.path.join(RESULTS, "responses_*.json"))):
        m = OLD.match(os.path.basename(path))
        if not m:
            continue  # already new-format (has a dataset tag)
        providers, n, seed = m.group(1), int(m.group(2)), int(m.group(3))
        blob = json.load(open(path))
        data = {k: np.asarray(v) for k, v in blob["data"].items()}
        fp_file = S.data_fingerprint(data)
        regen = S.get_dataset("synthetic", n=n, seed=seed)
        regen["_coord"] = np.random.default_rng(7_000 + seed).integers(-15, 16, len(regen["id"]))
        if fp_file != S.data_fingerprint(regen):
            print(f"  SKIP {os.path.basename(path)} (fingerprint != regenerated synthetic)")
            skipped += 1
            continue
        blob["fingerprint"] = fp_file
        new = os.path.join(RESULTS, f"responses_{providers}_synthetic_n{n}_s{seed}.json")
        action = "WOULD migrate" if not apply else "migrated"
        print(f"  {action} {os.path.basename(path)} -> {os.path.basename(new)}  (fp={fp_file})")
        if apply:
            json.dump(blob, open(new, "w"), indent=2)
            if os.path.abspath(new) != os.path.abspath(path):
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"      wrote new file; could not remove old ({e}); delete it manually")
        migrated += 1
    print(f"\n{'APPLIED' if apply else 'DRY-RUN'}: {migrated} to migrate, {skipped} skipped")
    if not apply and migrated:
        print("Re-run with --apply to perform the migration.")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)

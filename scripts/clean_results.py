#!/usr/bin/env python3
"""Tidy results/: keep the real paper results, remove test/pilot/mock scratch.

KEEPERS
  - synthetic DGP sweeps:
      r1_fp.csv  r2_power.csv  r3_separability.csv  table3_methods.csv
  - real-LLM 15-seed runs (anthropic-openai, n1000, S15) — synthetic & taiwan:
      real_llm_{aggregate,perseed}_anthropic-openai_{synthetic,taiwan}_n1000_x{reference,features}_S15.csv
      responses_anthropic-openai_{synthetic,taiwan}_n1000_s*.json
  - any taiwan run at other seed counts (real_llm_*taiwan*_S* + its responses)

REMOVED (scratch): every mock-mock_* file; n20/n120/n150/n200/n300/n8000 pilots;
single-seed _S1 pilots; the pre-pipeline real_llm_responses.json /
real_llm_results*.csv; and run_*.log. (_chunkcache/ is left alone — it is
gitignored resume scratch + the judge audit.)

NORMALIZES the synthetic reference run's filename to add the '_synthetic_' tag
(it predates dataset tagging) so synthetic & taiwan files line up.

Usage:  python scripts/clean_results.py [--apply]      (default: dry-run)
"""
from __future__ import annotations

import glob
import os
import re
import sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SWEEPS = {"r1_fp.csv", "r2_power.csv", "r3_separability.csv", "table3_methods.csv"}

_KEEP = [
    re.compile(r"^real_llm_(aggregate|perseed)_anthropic-openai_(synthetic_|taiwan_)?n1000_x(reference|features)_S15\.csv$"),
    re.compile(r"^real_llm_(aggregate|perseed)_anthropic-openai_taiwan_n1000_x(reference|features)_S\d+\.csv$"),
    re.compile(r"^responses_anthropic-openai_(synthetic|taiwan)_n1000_s\d+\.json$"),
]
_RENAME = re.compile(r"^real_llm_(aggregate|perseed)_anthropic-openai_n1000_x(reference|features)_S15\.csv$")


def is_keeper(name: str) -> bool:
    return name in SWEEPS or any(p.match(name) for p in _KEEP)


def rename_target(name: str):
    m = _RENAME.match(name)
    return (f"real_llm_{m.group(1)}_anthropic-openai_synthetic_n1000_x{m.group(2)}_S15.csv"
            if m else None)


def main(apply: bool) -> None:
    keep, rename, remove = [], [], []
    for path in sorted(glob.glob(os.path.join(R, "*"))):
        if os.path.isdir(path):
            continue
        name = os.path.basename(path)
        tgt = rename_target(name)
        if tgt:
            rename.append((name, tgt))
        elif is_keeper(name):
            keep.append(name)
        else:
            remove.append(name)

    print(f"KEEP   ({len(keep)}):")
    for n in keep:
        print(f"   {n}")
    print(f"\nRENAME ({len(rename)}):")
    for a, b in rename:
        print(f"   {a}\n      -> {b}")
    print(f"\nREMOVE ({len(remove)}):")
    for n in remove:
        print(f"   {n}")

    if apply:
        for a, b in rename:
            os.replace(os.path.join(R, a), os.path.join(R, b))
        for n in remove:
            try:
                os.remove(os.path.join(R, n))
            except OSError as e:
                print(f"   could not remove {n}: {e}")
        print(f"\nAPPLIED: renamed {len(rename)}, removed {len(remove)}, kept {len(keep)}")
    else:
        print(f"\nDRY-RUN: would rename {len(rename)}, remove {len(remove)}, keep {len(keep)}."
              "  Re-run with --apply (on your Mac) to execute.")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)

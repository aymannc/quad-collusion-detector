#!/usr/bin/env bash
# =============================================================================
# QUAD collusion detector — LOCAL (offline) run.
# No API keys, no cost. Validates the whole pipeline and produces the synthetic
# results, then runs the real-LLM PoC with the offline MOCK provider.
#
#   [1] smoke tests
#   [2] synthetic sweeps  (R1 false-positives, R2 power, R3 separability,
#                          head-to-head incl. the nonlinear regime)
#   [3] mock real-LLM PoC (M1/M2/M3 fire-rates with CIs)
#
# Usage:
#   bash scripts/run_local.sh            # full config
#   QUICK=1 bash scripts/run_local.sh    # fast smoke config (~1 min)
#   SYN_SEEDS=100 POC_SEEDS=10 bash scripts/run_local.sh   # override anything
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ---- config (override via environment) ----
if [ "${QUICK:-0}" = "1" ]; then
  SYN_SEEDS="${SYN_SEEDS:-20}";  SYN_N="${SYN_N:-150}";  SYN_NPERM="${SYN_NPERM:-60}"
  POC_N="${POC_N:-200}";         POC_SEEDS="${POC_SEEDS:-3}";  POC_NPERM="${POC_NPERM:-80}"
else
  SYN_SEEDS="${SYN_SEEDS:-200}"; SYN_N="${SYN_N:-500}";  SYN_NPERM="${SYN_NPERM:-200}"
  POC_N="${POC_N:-1000}";        POC_SEEDS="${POC_SEEDS:-15}"; POC_NPERM="${POC_NPERM:-200}"
fi

# ---- environment ----
set +u
[ -f "$REPO_DIR/../.venv/bin/activate" ] && source "$REPO_DIR/../.venv/bin/activate"
set -u
PY="$(command -v python || command -v python3)"
export PYTHONPATH="$REPO_DIR"
echo "==> python : $PY"
echo "==> repo   : $REPO_DIR"
"$PY" -c "import numpy, scipy" 2>/dev/null || "$PY" -m pip install numpy scipy --quiet

# ---- run ----
echo
echo "==> [1/3] smoke tests"
"$PY" tests/test_smoke.py

echo
echo "==> [2/3] synthetic sweeps  (seeds=$SYN_SEEDS  N=$SYN_N  n_perm=$SYN_NPERM)"
"$PY" -m experiments.run_synthetic --seeds "$SYN_SEEDS" --n "$SYN_N" --n-perm "$SYN_NPERM"

echo
echo "==> [3/3] mock real-LLM PoC  (N=$POC_N  seeds=$POC_SEEDS  — offline, no cost)"
"$PY" -m experiments.run_real_llm --provider mock --n "$POC_N" --seeds "$POC_SEEDS" --n-perm "$POC_NPERM"

echo
echo "==> DONE (offline). Results + logs in: $REPO_DIR/results/"
ls -1 "$REPO_DIR"/results/*.csv 2>/dev/null || true

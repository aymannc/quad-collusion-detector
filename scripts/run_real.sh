#!/usr/bin/env bash
# =============================================================================
# QUAD collusion detector — REAL run (queries live LLMs).
# Needs API keys; costs a few dollars. Two model families = stronger
# independence test.
#
#   [1] preflight   — verify keys/models work (no experiment cost)
#   [2] real PoC    — reference conditioning; fire-rates with 95% CIs
#   [3] ablation    — features conditioning, REUSES cached responses (free)
#
# Setup:
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   export OPENAI_API_KEY="sk-..."
#   bash scripts/run_real.sh
#
# Override anything via environment, e.g.:
#   MODEL_A=claude-opus-4-8 MODEL_B=gpt-5.5 N=2000 SEEDS=20 bash scripts/run_real.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ---- config (override via environment) ----
PROVIDER_A="${PROVIDER_A:-anthropic}";  MODEL_A="${MODEL_A:-claude-haiku-4-5}"   # generic, non-thinking
PROVIDER_B="${PROVIDER_B:-openai}";     MODEL_B="${MODEL_B:-gpt-4o}"             # generic, non-thinking
N="${N:-1000}"; SEEDS="${SEEDS:-15}"; CHUNK="${CHUNK:-50}"; NPERM="${NPERM:-200}"
PRICE_IN="${PRICE_IN:-1.75}"; PRICE_OUT="${PRICE_OUT:-7.5}"   # rough blend of Haiku + gpt-4o
export QUAD_LLM_RETRIES="${QUAD_LLM_RETRIES:-5}"   # auto-retry transient API errors (rate limit/timeout/5xx)

# ---- require API keys up front ----
: "${ANTHROPIC_API_KEY:?Set it first:  export ANTHROPIC_API_KEY=sk-ant-...}"
: "${OPENAI_API_KEY:?Set it first:  export OPENAI_API_KEY=sk-...}"

# ---- environment ----
set +u
[ -f "$REPO_DIR/../.venv/bin/activate" ] && source "$REPO_DIR/../.venv/bin/activate"
set -u
PY="$(command -v python || command -v python3)"
export PYTHONPATH="$REPO_DIR"
echo "==> python : $PY"
echo "==> models : A=$PROVIDER_A:$MODEL_A   B=$PROVIDER_B:$MODEL_B"
echo "==> config : N=$N seeds=$SEEDS chunk=$CHUNK n_perm=$NPERM"
"$PY" -c "import numpy, scipy" 2>/dev/null || "$PY" -m pip install numpy scipy --quiet
"$PY" -c "import anthropic, openai" 2>/dev/null || "$PY" -m pip install anthropic openai --quiet

COMMON=(--provider-a "$PROVIDER_A" --model-a "$MODEL_A"
        --provider-b "$PROVIDER_B" --model-b "$MODEL_B"
        --n "$N" --seeds "$SEEDS" --chunk "$CHUNK" --n-perm "$NPERM")
EXTRA=()
[ "${FRESH:-0}" = "1" ] && EXTRA+=(--fresh)   # FRESH=1 ignores cached responses

echo
echo "==> [1/3] preflight (verify keys/models; aborts here if anything is wrong)"
"$PY" -m experiments.run_real_llm --check-only "${COMMON[@]}"

echo
echo "==> [2/3] real PoC  (reference conditioning)"
"$PY" -m experiments.run_real_llm "${COMMON[@]}" "${EXTRA[@]}" --price-in "$PRICE_IN" --price-out "$PRICE_OUT"

echo
echo "==> [3/3] features-conditioning ablation (reuses cached responses — no extra cost)"
"$PY" -m experiments.run_real_llm "${COMMON[@]}" --x-mode features --use-cache

echo
echo "==> DONE. Aggregate tables:"
ls -1 "$REPO_DIR"/results/real_llm_aggregate_*.csv 2>/dev/null || true
echo "    (per-seed detail in results/real_llm_perseed_*.csv ; logs in results/run_*.log)"

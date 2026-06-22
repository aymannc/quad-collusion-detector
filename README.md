# QUAD Collusion Detector

A conditional-mutual-information (CMI) test for detecting **hidden coordination
among multiple LLM agents** — agreement that survives conditioning on the shared
input. Agents that merely react to the same input are conditionally independent;
any *residual* dependence given the input X is the fingerprint of collusion.

This repository contains the detector, a set of baseline comparators, the
synthetic experiments, and an automated real-LLM pipeline.

> **Scope.** The implemented and evaluated contribution is the CMI-based
> collusion detector. The broader QUAD oversight architecture is a *proposal*;
> the finance setting (credit decisions) is the *motivating* domain. The detector
> identifies *conditional statistical dependence*, which equals "collusion" only
> under the assumptions stated in the paper.

## Why CMI (and not correlation)?

A plain correlation test cannot tell "they agree because the case is obvious"
from "they agree because they coordinate" — both agents see the same input, so
they correlate either way. CMI conditions on that input X and asks whether
anything is *left over*.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy + scipy
# optional: pip install anthropic openai causal-learn pytest
```

## Quickstart (no API keys)

```bash
# synthetic experiments, fast smoke config
python -m experiments.run_synthetic --quick

# automated real-LLM experiment with the offline mock provider
python -m experiments.run_real_llm --provider mock --n 300
```

## Reproduce the paper

```bash
# full synthetic sweeps (R1 false positives, R2 power, R3 separability,
# Table 3 head-to-head incl. the nonlinear regime)
python -m experiments.run_synthetic --seeds 200 --n 500

# add the kernel-dependence baseline
python -m experiments.run_synthetic --table3-only --seeds 200 --use-hsic
```

Outputs are written to `results/` as CSV.

## Real LLMs (cross-provider)

Defaults use generic, non-thinking models — Agent A = Anthropic
`claude-haiku-4-5`, Agent B = OpenAI `gpt-4o`, N=1000, 15 seeds, chunked at
50/call, conditioning on the reference statistic — so a full real run is just:

```bash
export ANTHROPIC_API_KEY=...   OPENAI_API_KEY=...
python -m experiments.run_real_llm --check-only          # verify keys first
python -m experiments.run_real_llm --price-in 3 --price-out 15
```

It reports a per-condition, per-method **fire-rate with Wilson 95% CIs**. Two
different model families give a stronger independence test. Override any default
(`--model-a`, `--n`, `--seeds`, `--chunk`, `--x-mode`, ...); use `--provider mock`
for an offline dry run.

**Rate limits.** Requests are paced proactively to stay under your provider's
limit (so you don't hit 429 storms). Defaults assume a common gpt-4o tier
(30k TPM / 500 RPM); set `QUAD_OPENAI_TPM` / `QUAD_OPENAI_RPM` (and
`QUAD_ANTHROPIC_TPM` / `_RPM`) to match your tier — e.g. `QUAD_OPENAI_TPM=200000`
for gpt-4o-mini to run much faster. `0` disables pacing.

Every run collects fresh responses and saves them to
`results/responses_<config>.json`; pass `--use-cache` to reuse them (e.g. a
`--x-mode features` ablation at no API cost).

## Real dataset (German Credit)

By default the applicants are synthetic. To run on a **real** credit dataset
(UCI Statlog German Credit, 1,000 applicants), download it once into `data/`:

```bash
mkdir -p data
# direct file (classic UCI mirror) — this is the 21-column 'german.data':
curl -L -o data/german.data \
  https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
# verify you got the CLASSIC file — this MUST print 21:
awk 'NR==1{print NF}' data/german.data
# if it does NOT print 21, use the current UCI zip and extract german.data into data/:
#   curl -L -o /tmp/german.zip \
#     https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip
#   unzip -o /tmp/german.zip -d data/ && awk 'NR==1{print NF}' data/german.data   # -> 21
```

The loader needs the **classic 21-column `german.data`** (20 attributes + target,
whitespace-delimited, no header). It maps them to the official schema and uses the
7 numeric attributes as features (`duration_months, credit_amount,
installment_rate_pct, residence_since, age, existing_credits, num_dependents`),
dropping the target. Other variants (e.g. the 24-column `german.data-numeric`) are
rejected with a clear message — get the 21-column file.

Then switch the data source with `--dataset german` (or `DATASET=german` for the
script):

```bash
# real-credit run (gpt-4o-mini keeps it cheap and within rate limits):
DATASET=german MODEL_B=gpt-4o-mini bash scripts/run_real.sh
# or directly:
python -m experiments.run_real_llm --dataset german --n 1000 --price-in 3 --price-out 15
```

`--dataset-path` (default `data/german.data`) points at the file. German Credit
has 1,000 rows, so use `--n 1000` (a smaller `--n` samples a subset). Everything
else — detector, baselines, judge — is identical; only the applicant features
change, and outputs are tagged `…_german_…` so they don't collide with synthetic
runs. Requires `pandas` (in requirements.txt).

## Real dataset (Taiwan Credit Card Default)

A more modern, cleaner alternative (2005, 30,000 clients, 23 plain numeric
features — credit limit, age, six months of payment history, bill/payment
amounts). Download it once into `data/`:

```bash
mkdir -p data
curl -L -o /tmp/taiwan.zip \
  https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip
unzip -o /tmp/taiwan.zip -d data/
# the file extracts as "default of credit card clients.xls"; rename it:
mv "data/default of credit card clients.xls" data/taiwan_credit.xls
pip install xlrd openpyxl                      # readers for the .xls
```

The loader reads the UCI `.xls` (its real header is the 2nd row) or any CSV
mirror, drops `ID` and the target, and uses the 23 numeric features. Run it with
`--dataset taiwan` (or `DATASET=taiwan`):

```bash
DATASET=taiwan MODEL_B=gpt-4o-mini bash scripts/run_real.sh
# or directly:
python -m experiments.run_real_llm --dataset taiwan --n 1000 --price-in 3 --price-out 15
```

`--dataset-path` defaults to `data/taiwan_credit.xls`. It has 30,000 rows, so
`--n 1000` samples 1,000 applicants. Outputs are tagged `…_taiwan_…`.

## What's in the box

| Path | Contents |
|------|----------|
| `quad/cmi.py` | Frenzel–Pompe CMI estimator + local-conditional permutation test |
| `quad/baselines.py` | Pearson, partial correlation, distance correlation, residual dCor, HSIC, KCI |
| `quad/dgp.py` | synthetic regimes: independent, colluding, honest-competent, **nonlinear collusion** |
| `quad/scenarios.py` | programmatic credit-application generator (scales N, full feature X) |
| `quad/llm_agents.py` | Anthropic + OpenAI + mock providers; M1/M2/M3 condition prompts |
| `quad/analysis.py` | run detector + baselines, build fire/no-fire tables |
| `experiments/` | `run_synthetic.py`, `run_real_llm.py` |
| `tests/test_smoke.py` | fast correctness checks |

## Baselines (reviewer-requested)

CMI is benchmarked against more than the (deliberately weak) unconditional
Pearson test: **partial correlation** (linear conditional), **distance
correlation** and its residual/conditional form (nonlinear), **HSIC** (kernel
dependence), and optional **KCI** (kernel conditional independence, via
`causal-learn`). On linear-Gaussian collusion, partial correlation matches CMI;
the included **nonlinear-collusion** regime is where linear tests fail and
CMI / KCI / residual-dCor still detect the coordination — which is what
justifies CMI over the cheap linear baseline.

## Tests

```bash
pip install pytest && pytest -q          # or: python tests/test_smoke.py
```

## Citation

If you use this code, please cite the QUAD paper (see `CITATION` once the DOI is
assigned).

## License

MIT — see [LICENSE](LICENSE).

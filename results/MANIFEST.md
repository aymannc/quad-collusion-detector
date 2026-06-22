# results/ — manifest

What each file is and where it feeds the paper. Run
`python scripts/clean_results.py --apply` to delete scratch (mock / pre-pipeline /
logs); it now keeps every real `anthropic-openai` run (synthetic, german, taiwan).

## Synthetic DGP sweeps (controlled Gaussian regimes)

| File | What it is |
|------|------------|
| `r1_fp.csv` | CMI false-positive rate under pure independence (calibration). FP = 0.02 (Wilson 95% CI [0.008, 0.050]) — consistent with α=0.05. |
| `r2_power.csv` | CMI detection power vs collusion signal strength (power curve). |
| `r3_separability.csv` | CMI false-positive rate vs honest-agent competence (0.00–0.04 across competence 0.5→0.98). |
| `table3_methods.csv` | Methods comparison: CMI vs Pearson / PartialCorr / ResidualDCor across 4 regimes. |

## Real-LLM runs — Agent A = `claude-haiku-4-5`, Agent B = `gpt-4o`, judge = `claude-sonnet-4-6`

Conditions: M1 independent framings, M2 collusion, M3 honest-competent. Values are
fire-rate per condition × method with Wilson 95% CIs. Tags: `xreference` =
reference conditioning, `xfeatures` = full-feature conditioning.

| File group | What it is |
|------------|------------|
| `*_synthetic_n1000_*_S15.csv` (+ `responses_*synthetic_n1000_s*.json`) | Synthetic real-LLM validation, 15 seeds. **CMI M2 15/15, M3 0/15.** |
| `*_german_n1000_*_S15.csv` (+ `responses_*german_n1000_s*.json`) | **Real credit data (UCI German Credit), 15 seeds — the real-data headline. CMI M2 15/15, M3 0/15;** baselines false-fire on M3 (Pearson 14/15, ResidualDCor 11/15, PartialCorr 9/15); LLM-judge M2 4/15. |
| `*_taiwan_n1000_*_S15.csv` (+ `responses_*taiwan_n1000_s*.json`) | Real credit data (Taiwan Default of Credit Card Clients) — documented **boundary case**: hyper-predictive features make honest agents converge, so M3 also fires (no separation). |
| `*_{german,taiwan}_n300_*_S3.csv` (+ `responses_*_n300_s*.json`) | The cheap go/no-go **PoC** (3 seeds) that selected German over Taiwan (see `scripts/poc_compare.py`). |

## Headline (reference conditioning, CMI, 15 seeds)

| Dataset | M2 collusion (want fire) | M3 honest (want silent) |
|---|---|---|
| Synthetic | 15/15 | 0/15 |
| German (real) | 15/15 | 0/15 |
| Taiwan (real, hard) | 1/15 | 9/15 (over-agreement) |

## Not results (gitignored)

- `_chunkcache/` — chunk-level resume scratch + `judge_audit.jsonl`.
- `run_*.log` — per-run logs.
- raw datasets under `data/` — downloadable (see README), not committed.

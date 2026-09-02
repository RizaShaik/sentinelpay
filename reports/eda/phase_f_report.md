# SentinelPay -- Phase F Causal Target-Derived Historical Fraud-Rate Feature Report

**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_f_report` from `reports/eda/phase_f_results.json`** -- every number below is read from that file; re-running `python -m sentinelpay.eda.run_phase_f` regenerates both together.

## Scope (what Phase F is, and deliberately is not)

Phase F is a minimal, causal, fold-safe TARGET-DERIVED historical
fraud-rate feature for `payment_proxy_key` ONLY -- the one key D.1 evidenced
as suitable (87.05% row coverage, 78.66% sufficiency, partition-stable,
dominant-group robust; `device_proxy_key` failed D.1's row-coverage bar at
20.30%). Explicitly out of scope:

- **No `device_proxy_key`, no E.1/E.2 relationship-node keys, no generic
  multi-key framework.** This module is intentionally narrow.
- **No new causal primitive.** `sentinelpay.data.history.prior_group_amount_stats`
  is reused UNCHANGED (called with `amount_col="isFraud"`) -- it remains
  fully generic and is not modified; the target dependency is introduced
  only in `sentinelpay.target_history`, a new, explicitly isolated module.
- **Explicit source/recipient eligibility, not incidental filtering.**
  `sentinelpay.target_history.build_eligible_pools` constructs the `train`
  and `validation` pools explicitly; `compute_prior_fraud_rate` itself
  RAISES if given a pool containing any partition outside the caller's
  declared `allowed_source_partitions`. `embargo_1`/`embargo_2`/`holdout`
  are never label sources for anyone and never receive a computed feature
  value in this phase, regardless of chronological order.
- **No hyperparameter tuning.** `SMOOTHING_K` and the fraud-rate bucket
  edges below are fixed BEFORE the validation-only evaluation runs, and are
  never reselected from its results.
- **One-time EDA evaluation only.** No production feature, score,
  threshold, `configs/detection.yaml`-style config, or further phase is
  added by this script regardless of its result.

## 1. Split configuration (unchanged from Phase B/C/D/D.1/E.1/E.2)

| partition | start_day | end_day |
|---|---|---|
| train | 1 | 130 |
| embargo_1 | 131 | 137 |
| validation | 138 | 160 |
| embargo_2 | 161 | 167 |
| holdout | 168 | 182 |

## 2. Holdout sealing

- Total rows loaded (train_transaction.csv): **590,540**.
- Rows filtered to `train`/`embargo_1`/`validation`/`embargo_2` **before** any key-building or history computation: **549,899**.
- Holdout rows excluded, never touched: **40,641**.
- `payment_proxy_key` present on **478,702** / 549,899 development rows (**71,197** excluded, missing a key component).
- `isFraud` IS loaded as a feature input in this phase (the first phase where that's true) -- see scope section above for what remains unchanged (no out-of-order use, no hyperparameter tuning).

## 3. Explicit eligible pools (source/recipient contract)

- `train_pool`: **386,407** rows, partitions present: `['train']`.
- `validation_pool`: **444,213** rows, partitions present: `['train', 'validation']`.
- `embargo_1`/`embargo_2`/`holdout` are absent from both pools BY CONSTRUCTION (`sentinelpay.target_history.build_eligible_pools`), not by incidental filtering.

## 4. Fixed hyperparameters (declared before validation-only evaluation)

- `SMOOTHING_K` = **20.0** (project-consistency value, NOT a derived statistical equivalence to any other constant -- see `sentinelpay.target_history` module docstring).
- `SUFFICIENT_HISTORY_THRESHOLD` (diagnostic only) = **5**.
- `fraud_rate_bucket_edges` (from TRAIN's own smoothed-rate p25/p50/p75/p90, never validation's): `[0.009569809988147202, 0.0162331935269395, 0.021787118633024096, 0.035637045359270085]` -> ['rate_lt_p25', 'rate_p25_to_p50', 'rate_p50_to_p75', 'rate_p75_to_p90', 'rate_ge_p90']

## 5. Feature descriptive summary (non-circular: train uses only train's own prior history)

### train (n_rows=386,407)

**payment_proxy_prior_fraud_rate_smoothed:**

n_rows=386,407 | n_nan=1

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 0.000000 | 0.009570 | 0.016233 | 0.021787 | 0.035637 | 0.161262 | 0.770139 | 0.021592 |

**payment_proxy_prior_fraud_rate_raw:**

n_rows=386,407 | n_nan=34,207

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 0.000000 | 0.000000 | 0.000000 | 0.015094 | 0.041667 | 0.375000 | 1.000000 | 0.021163 |

**sufficient_target_history:** n_true=293,901 | n_false=92,506 | pct_true=76.0600%

**global_cold_start:** n_true=1 | n_false=386,406 | pct_true=0.0003%

### validation (n_rows=57,806)

**payment_proxy_prior_fraud_rate_smoothed:**

n_rows=57,806 | n_nan=0

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 0.000460 | 0.008683 | 0.015460 | 0.024206 | 0.040328 | 0.179754 | 0.742650 | 0.022719 |

**payment_proxy_prior_fraud_rate_raw:**

n_rows=57,806 | n_nan=1,900

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 0.000000 | 0.000000 | 0.000000 | 0.022248 | 0.045226 | 0.285714 | 1.000000 | 0.021816 |

**sufficient_target_history:** n_true=51,590 | n_false=6,216 | pct_true=89.2468%

**global_cold_start:** n_true=0 | n_false=57,806 | pct_true=0.0000%


## 6. Strict validation-only evaluation (isFraud, descriptive evidence only)

`n_validation_rows`: **57,806** | overall fraud rate: **0.022385**

- `roc_auc_smoothed_rate_vs_isFraud`: **0.7850** (n=57,806)
- `roc_auc_raw_rate_vs_isFraud`: **0.7998** (n=55,906 -- raw rate is NaN at per-key cold start, a smaller, DIFFERENT population than the smoothed-rate AUC above; not directly comparable row-for-row).

**Fraud rate by fixed `payment_proxy_prior_fraud_rate_smoothed` bucket:**

| bucket | n_rows | fraud_rate |
|---|---|---|
| rate_lt_p25 | 16243 | 0.006403 |
| rate_p25_to_p50 | 13956 | 0.008742 |
| rate_p50_to_p75 | 8883 | 0.011933 |
| rate_p75_to_p90 | 11833 | 0.023916 |
| rate_ge_p90 | 6891 | 0.098534 |

**Fraud rate by `sufficient_target_history`:**

| sufficient_target_history | n_rows | fraud_rate |
|---|---|---|
| True | 51590 | 0.021671 |
| False | 6216 | 0.028314 |


**sufficient_target_history coverage (validation):** n_true=51,590 | n_false=6,216 | pct_true=89.2468%

**global_cold_start coverage (validation):** n_true=0 | n_false=57,806 | pct_true=0.0000%


## 7. No production feature, scoring model, or further phase

Nothing in this report is a production feature, score, threshold, `configs/detection.yaml`-style config, or further phase work -- this is a one-time EDA measurement + validation-only evaluation only. `SMOOTHING_K` and the fraud-rate bucket edges above were fixed before this evaluation ran and were not reselected from its results.

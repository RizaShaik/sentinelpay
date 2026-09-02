# SentinelPay -- Phase G Development-Only Model Integration & Ablation Report

**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_g_report` from `reports/eda/phase_g_results.json`** -- every number below is read from that file; re-running `python -m sentinelpay.eda.run_phase_g` regenerates both together.

## Scope (what Phase G is, and deliberately is not)

Phase G is a development-only model integration and ablation pass over the
B0-F2 baseline ladder, headline comparison B2 vs. F2. Phase F is consumed
FROZEN and VERBATIM -- no re-tuning of `SMOOTHING_K`, no redefinition of
smoothing, no new target-history logic (`sentinelpay.target_history` and
`sentinelpay.eda.run_phase_f` are both untouched). Phase C and Phase D are
likewise consumed unmodified. Explicitly out of scope:

- **No feature-matrix persistence.** `data/processed/*.parquet` is NOT
  written -- matrices are built in memory, once per run.
- **No hyperparameter tuning.** Logistic Regression only, library defaults
  (`LOGREG_MAX_ITER` is a solver-convergence budget, not a modeling
  choice); no tree/boosting model in this pass.
- **No new phase started.** This is a one-time development-only evaluation.
- **Holdout completely untouched** -- never loaded from disk by this
  script.

See `sentinelpay.model_features`'s module docstring for the full, explicit
per-row leakage contract (including the intentional Phase D vs. Phase F
embargo-eligibility asymmetry) every feature block below satisfies.

## 1. Split configuration (unchanged from Phase B/C/D/D.1/E.1/E.2/F)

| partition | start_day | end_day |
|---|---|---|
| train | 1 | 130 |
| embargo_1 | 131 | 137 |
| validation | 138 | 160 |
| embargo_2 | 161 | 167 |
| holdout | 168 | 182 |

## 2. Holdout sealing

- Total rows loaded (train_transaction.csv): **590,540**.
- Rows filtered to development partitions **before** any key-building or feature assembly: **549,899**.
- Holdout rows excluded, never touched: **40,641**.
- `payment_proxy_key` present on **478,702** development rows (**71,197** excluded).
- Model rows: `train`=**386,407**, `validation`=**57,806**.

## 3. Fixed ladder feature schema (identical order for train and validation)

- **B1** (5 features): `['amt_log1p', 'amt_decimal_part', 'dt_hour_of_day', 'dt_day_of_week', 'has_identity']`

- **B2** (12 features): `['amt_log1p', 'amt_decimal_part', 'dt_hour_of_day', 'dt_day_of_week', 'has_identity', 'prior_median', 'prior_mad', 'prior_count_in_window', 'modified_zscore', 'flag_insufficient_history', 'flag_zero_mad', 'flag_scored_outlier']`

- **F1** (14 features): `['amt_log1p', 'amt_decimal_part', 'dt_hour_of_day', 'dt_day_of_week', 'has_identity', 'prior_median', 'prior_mad', 'prior_count_in_window', 'modified_zscore', 'flag_insufficient_history', 'flag_zero_mad', 'flag_scored_outlier', 'payment_proxy_prior_fraud_rate_smoothed', 'global_cold_start']`

- **F2** (15 features): `['amt_log1p', 'amt_decimal_part', 'dt_hour_of_day', 'dt_day_of_week', 'has_identity', 'prior_median', 'prior_mad', 'prior_count_in_window', 'modified_zscore', 'flag_insufficient_history', 'flag_zero_mad', 'flag_scored_outlier', 'payment_proxy_prior_fraud_rate_smoothed', 'global_cold_start', 'sufficient_target_history']`


## 4. Ladder results (validation-only)

| step | n_features | converged | roc_auc | pr_auc |
|---|---|---|---|---|
| B0 | 0 | True | 0.500000 | 0.022385 |
| B1 | 5 | True | 0.677100 | 0.055462 |
| B2 | 12 | True | 0.678565 | 0.052308 |
| F1 | 14 | True | 0.793937 | 0.168084 |
| F2 | 15 | True | 0.793913 | 0.168045 |


## 5. Graduation gates (fixed before this evaluation ran)

- **Gate 1** -- relative PR-AUC lift, F2 >= B2 x 1.1: **True** (actual lift: 3.2126x)
- **Gate 2** -- ROC-AUC(F2) >= ROC-AUC(B2): **True**
- **Gate 3** -- bootstrap 95% CI lower bound of [PR-AUC(F2) - PR-AUC(B2)] > 0: **True** (mean_delta=0.116355, ci=[0.098295, 0.136661], n_resamples_used=1000/1000, seed=20260101)
- **Gate 4** -- PR-AUC(F1) > PR-AUC(B2), deterministic: **True**
- Reported (not a gate): B2 <= F1 <= F2 monotonic in PR-AUC: **False**

### ALL GATES PASS: **True**


## 6. No persistence, no new phase

No `data/processed/*.parquet` was written. No production feature, score, threshold, config, or further phase is started by this script regardless of the result above -- this is a one-time development-only evaluation.

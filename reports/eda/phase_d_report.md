# SentinelPay -- Phase D Behavioral-Change Detector Report

**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_d_report` from `reports/eda/phase_d_results.json`** -- every number below is read from that file; re-running `python -m sentinelpay.eda.run_phase_d` regenerates both together.

## Scope (what Phase D is, and deliberately is not)

Phase D is the first real per-`payment_proxy_key` behavioral-change
detector, built on D.1's evidence-based grouping-key recommendation. Explicit
hard non-target boundary, enforced by tests, not just convention:

- **`sentinelpay.data.history` and `sentinelpay.detection` never import,
  accept, or reference `isFraud` (or any target column)** in any function
  signature or computation. Every score, flag, cold-start decision, and
  zero-MAD decision is computed strictly from `TransactionID`,
  `TransactionDT`, `TransactionAmt`, and `payment_proxy_key` columns.
- **`isFraud` is read in exactly one place**: the validation-only evaluation
  section below, which runs strictly AFTER every score/flag already exists,
  reads `isFraud` only for `validation`-partition rows, and never writes
  anything back into a score, a flag, or `configs/detection.yaml`.
- **No hyperparameter is selected, tuned, or changed based on validation
  `isFraud` results** -- every value in the hyperparameter table below was
  fixed before this evaluation ran (see `configs/detection.yaml`).
- **`device_proxy_key` is not used** -- D.1 found it unsuitable (row
  coverage 20.30% < the 50% bar).
- **No EWMA, CUSUM/change-point, or Isolation Forest** -- median/MAD
  (Iglewicz & Hoaglin modified z-score) is the sole method this phase.
- **No target encoding, no historical-fraud-rate feature, no coordinated-ring
  detection (Phase E), no `data/processed/*.parquet` persistence.**
- **`configs/split.yaml` boundaries and embargo widths are unchanged.**

## 1. Split configuration (unchanged from Phase B/C/D.1)

| partition | start_day | end_day |
|---|---|---|
| train | 1 | 130 |
| embargo_1 | 131 | 137 |
| validation | 138 | 160 |
| embargo_2 | 161 | 167 |
| holdout | 168 | 182 |

## 2. Detector hyperparameters (fixed before any evaluation)

All five values are loaded from `configs/detection.yaml`, never selected, tuned, or changed based on validation `isFraud` results.

| parameter | value | justification |
|---|---|---|
| `min_history_for_score` | 5 | Reuses D.1's `DECISION_THRESHOLD` -- the pre-declared, already-validated (partition-stable, dominant-group-robust) "enough causal history" bar for `payment_proxy_key`. |
| `window_size_events` | 20 | Reuses D.1's largest pre-declared threshold bucket (>= 20), already measured as majority-stable for `payment_proxy_key` across partitions and dominant-group exclusion. |
| `modified_zscore_scale_constant` | 0.6745 | Standard Iglewicz & Hoaglin (1993) modified z-score consistency constant -- a mathematical property of the statistic, not tuned to this dataset. |
| `modified_zscore_threshold` | 3.5 | Standard, widely cited Iglewicz & Hoaglin outlier cutoff for the modified z-score. |
| `zero_mad_epsilon` | 1e-09 | Numerical-safety floor only (avoids division blow-up when prior amounts are near-identical), not a detection-sensitivity knob. |

## 3. Holdout sealing

- Total rows loaded (`train_transaction.csv`): **590,540**.
- Rows filtered to `train`/`embargo_1`/`validation`/`embargo_2` (`sentinelpay.data.split.DEVELOPMENT_PARTITIONS`) **before** `build_group_key`/`compute_behavioral_change_score` are ever called: **549,899**.
- Holdout rows excluded, never touched by group-key, history, or score computation: **40,641**.
- Of the development rows, **478,702** have a valid `payment_proxy_key` (71,197 excluded, missing a key component).
- `isFraud` is never read while building the detector; it is read only in section 5 below, only for `validation`-partition rows.

## Embargo semantics: continuous history, no partition-boundary reset

The detector's inputs are non-target aggregates (median/MAD of `amt_log1p`,
event counts), so history is computed once over the full concatenated
`train`+`embargo_1`+`validation`+`embargo_2` frame, sorted causally by
`TransactionDT` -- partition labels are never used to gate, reset, or segment
the history computation itself, only to group this report and the
validation-only evaluation. `embargo_1` rows legitimately contribute to
`validation` rows' windows, exactly as Phase C established for non-target
historical aggregates (see reports/eda/phase_c_report.md). Holdout rows
remain excluded before any content computation -- see Holdout sealing below.

## 4. Score coverage by partition (non-target)

| partition | n_rows | pct_insufficient_history | pct_zero_mad | pct_scored_normal | pct_scored_outlier |
|---|---|---|---|---|---|
| train | 386407 | 23.9400 | 1.5556 | 70.6918 | 3.8126 |
| embargo_1 | 16386 | 11.0155 | 1.8186 | 83.1991 | 3.9668 |
| validation | 57806 | 10.2982 | 1.8199 | 83.6678 | 4.2141 |
| embargo_2 | 18103 | 10.5397 | 3.8281 | 81.3235 | 4.3087 |


**`abs(modified_zscore)` distribution among scored rows (all development partitions):**

| n_scored_rows | abs_modified_zscore_p50 | abs_modified_zscore_p75 | abs_modified_zscore_p90 | abs_modified_zscore_p99 |
|---|---|---|---|---|
| 368476 | 0.7462 | 1.4132 | 2.4510 | 8.4561 |


## 5. `validation`-only target-association evaluation (diagnostic only -- NOT a feature, NOT used to select any hyperparameter)

- `validation` rows: **57,806**. Flag breakdown: `insufficient_history` 10.30%, `zero_mad` 1.82%, `scored_normal` 83.67%, `scored_outlier` 4.21%.
- Scored rows (non-`NaN` score) used for the metrics below: **50,801**.

**Fraud rate by `modified_zscore` decile (scored `validation` rows only):**

| score_decile | n_rows | fraud_rate |
|---|---|---|
| (-1501.826, -1.385] | 5081 | 0.0146 |
| (-1.385, -0.855] | 5080 | 0.0175 |
| (-0.855, -0.579] | 5080 | 0.0189 |
| (-0.579, -0.248] | 5080 | 0.0199 |
| (-0.248, 0.0] | 6175 | 0.0215 |
| (0.0, 0.245] | 3985 | 0.0203 |
| (0.245, 0.647] | 5080 | 0.0248 |
| (0.647, 1.063] | 5080 | 0.0258 |
| (1.063, 1.934] | 5080 | 0.0256 |
| (1.934, 464.237] | 5080 | 0.0268 |


**Fraud rate: `scored_outlier` vs. `scored_normal` (scored `validation` rows only):**

| flag | n_rows | fraud_rate |
|---|---|---|
| scored_outlier | 2436 | 0.0250 |
| scored_normal | 48365 | 0.0214 |


**ROC-AUC of `abs(modified_zscore)` vs. `isFraud`** (scored `validation` rows only, n=50,801): **0.4992**.

These four diagnostics are reported once, for the fixed configuration in section 2 above. No threshold, window size, or constant is reselected from these results, and no sensitivity/sweep table against `isFraud` is produced in this phase.
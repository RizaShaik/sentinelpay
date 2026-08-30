# SentinelPay -- Phase C Non-Target Feature Foundation Report

**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_c_report` from `reports/eda/phase_c_results.json`** -- every number below is read from that file; re-running `python -m sentinelpay.eda.run_phase_c` regenerates both together.

## Scope (what Phase C is, and deliberately is not)

Phase C is a non-target, chronology-safe **feature foundation**, not a
broad model-ready feature expansion. Explicitly out of scope, per review of
the original (broader) Phase C proposal:

- **No target encoding of any kind.** Nothing below reads or depends on
  `isFraud`; `sentinelpay.features.build_feature_frame` does not accept a
  target column. A future target-derived historical feature is deferred
  until Phase D/E architecture (including embargo width) is settled.
- **No V1-V339 handling** -- no raw passthrough, no block aggregates.
- **No `data/processed/*_features.parquet` persistence** -- features are
  built in memory, validated, and reported only.
- **No production grouping-key selection.** `sentinelpay.data.history`'s
  historical utilities are generic and covered by unit tests only (synthetic
  group keys) -- see section below. `payment_proxy_key`/`device_proxy_key`/
  `ProductCD` are not used anywhere in Phase C.
- **One unified feature pipeline** -- a single `has_identity` indicator, not
  parallel with/without-identity feature sets.
- **`configs/split.yaml` boundaries and embargo widths are unchanged.**

## 1. Split configuration (unchanged from Phase B)

| partition | start_day | end_day |
|---|---|---|
| train | 1 | 130 |
| embargo_1 | 131 | 137 |
| validation | 138 | 160 |
| embargo_2 | 161 | 167 |
| holdout | 168 | 182 |

## 2. Holdout sealing

- Total rows loaded (single `train_transaction.csv`, spans all partitions): **590,540**.
- Rows filtered to `train`/`embargo_1`/`validation`/`embargo_2` (`sentinelpay.data.split.DEVELOPMENT_PARTITIONS`) **before** `build_feature_frame` is ever called: **549,899**.
- Holdout rows excluded, never touched by feature computation: **40,641**.
- `isFraud` is never loaded by `sentinelpay.eda.run_phase_c` -- only `TransactionID`, `TransactionDT`, `TransactionAmt` are read from `train_transaction.csv`.

## 3. Feature registry

| feature | source_columns | uses_target | temporal_dependency | description |
|---|---|---|---|---|
| amt_log1p | TransactionAmt | False | row-local | log1p(TransactionAmt) -- variance-stabilizing transform of a documented, always-populated field. |
| amt_decimal_part | TransactionAmt | False | row-local | TransactionAmt - floor(TransactionAmt); distinguishes round-dollar vs. exact-cent amounts. |
| dt_hour_of_day | TransactionDT | False | row-local | (TransactionDT // 3600) % 24 -- cyclical time-of-day bucket from the row's own timestamp only. |
| dt_day_of_week | TransactionDT | False | row-local | (TransactionDT // 86400) % 7 -- cyclical period-7 bucket; not a claimed calendar weekday (epoch undisclosed). |
| has_identity | TransactionID | False | structural-join | Whether TransactionID has a matching identity-file row; existence only, no identity-table value used. |


## 4. Feature summary by partition (train/embargo_1/validation/embargo_2 only)

| partition | feature | mean | std | pct_missing | n_rows |
|---|---|---|---|---|---|
| train | amt_log1p | 4.3844 | 0.9330 | 0.0000 | 445173 |
| train | amt_decimal_part | 0.3732 | 0.4316 | 0.0000 | 445173 |
| train | dt_hour_of_day | 13.8255 | 7.6763 | 0.0000 | 445173 |
| train | dt_day_of_week | 2.9301 | 2.0153 | 0.0000 | 445173 |
| train | has_identity | 0.2592 | 0.4382 | 0.0000 | 445173 |
| embargo_1 | amt_log1p | 4.3482 | 0.9580 | 0.0000 | 18589 |
| embargo_1 | amt_decimal_part | 0.4150 | 0.4422 | 0.0000 | 18589 |
| embargo_1 | dt_hour_of_day | 13.9315 | 7.4904 | 0.0000 | 18589 |
| embargo_1 | dt_day_of_week | 3.0242 | 2.0395 | 0.0000 | 18589 |
| embargo_1 | has_identity | 0.1856 | 0.3888 | 0.0000 | 18589 |
| validation | amt_log1p | 4.3920 | 0.9424 | 0.0000 | 65193 |
| validation | amt_decimal_part | 0.4159 | 0.4447 | 0.0000 | 65193 |
| validation | dt_hour_of_day | 13.9387 | 7.4007 | 0.0000 | 65193 |
| validation | dt_day_of_week | 3.2321 | 2.0986 | 0.0000 | 65193 |
| validation | has_identity | 0.1742 | 0.3793 | 0.0000 | 65193 |
| embargo_2 | amt_log1p | 4.3501 | 0.9606 | 0.0000 | 20944 |
| embargo_2 | amt_decimal_part | 0.3786 | 0.4350 | 0.0000 | 20944 |
| embargo_2 | dt_hour_of_day | 14.0019 | 7.2815 | 0.0000 | 20944 |
| embargo_2 | dt_day_of_week | 3.0025 | 1.9960 | 0.0000 | 20944 |
| embargo_2 | has_identity | 0.2610 | 0.4392 | 0.0000 | 20944 |


## Historical utilities (`sentinelpay.data.history`) -- not applied to real data in Phase C

Three generic, target-agnostic causal aggregation functions are implemented
and unit-tested (`tests/test_history.py`), but are **not called against real
data** in this phase -- there is no approved production grouping key yet
(see scope above), and demonstrating them against a real column (as Phase B
did for the day-bucket rollup utilities) is exactly what review of the
original proposal disallowed.

- `prior_group_count`, `prior_group_amount_stats`, `time_since_last_group_event`.
- **Guarantee**: for row i, only same-group rows with `TransactionDT`
  strictly less than row i's ever contribute to row i's result. Two rows
  sharing a group and an identical `TransactionDT` never see each other.
  Row order in the input never matters, and `TransactionID` is never used as
  a tie-breaker.
- **Test coverage**: tie handling, adversarial perturbation (mutating or
  adding a future row leaves every earlier row's result unchanged),
  row-order independence, and no-target-dependency are all asserted in
  `tests/test_history.py` against synthetic group keys only.

**Embargo note**: these are non-target aggregates (count/sum/mean of
`TransactionAmt`, recency), not fraud-rate statistics. `embargo_1` rows
chronologically precede `validation` rows, so they DO contribute to a
validation row's historical aggregate -- the 7-day embargo exists to guard a
*label-based* boundary decision against undocumented D-column lookback
windows (see reports/eda/phase_b_report.md section 3), not to blank out real
antecedent transaction content from non-target feature computation.
Excluding embargo rows would only make early-validation historical features
artificially sparse relative to real scoring-time behavior. This reasoning
does NOT extend to any future target-derived historical feature -- that is
exactly the case the embargo protects against, and none is implemented here.
Holdout rows are loaded into memory along with everything else (they share
`train_transaction.csv` with every other partition -- there is no way to read
the file without them), but are excluded via `assign_partition` immediately
after loading and before any content computation, including before
`build_feature_frame` is ever called (see section 2 above). So this question
does not arise for holdout: no content statistic, historical or otherwise, is
ever computed on a holdout row in Phase C.

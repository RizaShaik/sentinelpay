# SentinelPay -- Phase E.2 Union-Find Component Metrics Report (Measurement Pass)

**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_e2_report` from `reports/eda/phase_e2_results.json`** -- every number below is read from that file; re-running `python -m sentinelpay.eda.run_phase_e2` regenerates both together.

## Scope (what Phase E.2 is, and deliberately is not)

Phase E.2 is a minimal, non-target per-transaction Union-Find /
connected-component MEASUREMENT pass for exactly one relationship:
device_node <-> payment_node -- the only relationship whose Phase E.1
M5b frequency-adjusted overlap diagnostic cleared `LIFT_SIGNAL_THRESHOLD`
(lift_ratio 11.06x; see reports/eda/phase_e1_report.md section 5).
Explicitly out of scope:

- **No other relationship.** payment_email_purchaser and device_email_purchaser
  both stayed below `LIFT_SIGNAL_THRESHOLD` in E.1 and are not built here.
- **No scoring model, no flags, no `configs/detection.yaml`-style
  persistence.** This is a measurement pass only.
- **No target of any kind, except in exactly one place.** `isFraud` is never
  loaded or read in `sentinelpay.eda.component_analysis` or
  `sentinelpay.data.causal_components`, and every non-target quantity below
  (all four component metrics, plus the fan-out stratification variable) is
  fully computed before `isFraud` is read at all. The pre-declared
  fan-out-stratified diagnostic evaluation (section 6) is the ONE
  strictly-downstream, `validation`-partition-only place `isFraud` is read in
  this phase -- mirrors Phase D's `evaluate_validation_only` precedent
  exactly, and its fixed bin edges were declared from already-published
  non-target percentiles BEFORE this evaluation ever read `isFraud` (see
  section 6). This is a one-time EDA evaluation only: no production feature,
  score, threshold, config, or E.3 work is added regardless of its result.
- **No E.3 work of any kind.**

**Per-transaction metrics, computed at read time strictly from Union-Find
state built from earlier TransactionDT buckets only** (see
`sentinelpay.data.causal_components` for the full causal contract, including
why the actual Union-Find mutation for a bucket is deferred until every row
in that bucket has already been read):

- `device_component_size_total` -- size of the device_node's own component.
- `payment_component_size_total` -- size of the payment_node's own component.
- `endpoints_same_component` -- whether the two endpoints were already in
  the same component.
- `merged_component_size_total` -- the hypothetical size if this edge were
  unioned right now: the shared size if `endpoints_same_component`, else the
  sum of the two component sizes above. A current edge's two endpoints can
  already belong to two DIFFERENT causal components -- this metric (and the
  two component-size metrics it's built from) is why this phase reports
  both endpoints' state rather than only the component containing one of
  them.

## 1. Split configuration (unchanged from Phase B/C/D/D.1/E.1)

| partition | start_day | end_day |
|---|---|---|
| train | 1 | 130 |
| embargo_1 | 131 | 137 |
| validation | 138 | 160 |
| embargo_2 | 161 | 167 |
| holdout | 168 | 182 |

## 2. Holdout sealing

- Total rows loaded (train_transaction.csv joined to train_identity.csv): **590,540**.
- Rows filtered to `train`/`embargo_1`/`validation`/`embargo_2` (`sentinelpay.data.split.DEVELOPMENT_PARTITIONS`) **before** any component computation: **549,899**.
- Holdout rows excluded, never touched by relationship-frame or component computation: **40,641**.
- `isFraud` is never loaded by `sentinelpay.eda.run_phase_e2`.

## 3. device_node <-> payment_node relationship row coverage

**73,305** / 549,899 development rows (**13.33%**) have both `_device_node` and `_payment_node` present -- identical row coverage to Phase E.1's payment_device relationship pair (both directions share the same valid-row set).

## 4. Per-transaction component metrics -- overall descriptive summary

n_rows: **73,305**

**device_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 1 | 7247.0000 | 12107.0000 | 15810.0000 | 17900.0000 | 18997.9600 | 19090 | 11263.7107 |

**payment_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 1 | 52.0000 | 10427.0000 | 15298.0000 | 17733.0000 | 18988.0000 | 19090 | 9193.2417 |

**merged_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 2 | 7506.0000 | 12241.0000 | 15893.0000 | 17925.0000 | 18998.0000 | 19090 | 11446.3551 |

**endpoints_same_component:** n_true=53,999 | n_false=19,306 | pct_true=73.6635%


## 5. Per-transaction component metrics -- by partition

### partition: train (n_rows=64,130)

**device_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 1 | 6571.2500 | 11126.0000 | 14541.7500 | 16418.1000 | 17510.0000 | 17625 | 10305.9969 |

**payment_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 1 | 1.0000 | 9269.0000 | 13937.0000 | 16221.0000 | 17486.7100 | 17625 | 8251.9236 |

**merged_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 2 | 6807.0000 | 11241.5000 | 14602.0000 | 16443.0000 | 17512.0000 | 17625 | 10463.5438 |

**endpoints_same_component:** n_true=46,305 | n_false=17,825 | pct_true=72.2049%

### partition: embargo_1 (n_rows=1,340)

**device_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 1 | 17671.0000 | 17736.5000 | 17796.2500 | 17825.0000 | 17842.0000 | 17848 | 17339.0985 |

**payment_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 1 | 17643.0000 | 17716.0000 | 17791.0000 | 17824.0000 | 17842.0000 | 17848 | 15023.9425 |

**merged_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 2 | 17674.0000 | 17737.0000 | 17799.0000 | 17825.0000 | 17842.6100 | 17848 | 17657.4022 |

**endpoints_same_component:** n_true=1,112 | n_false=228 | pct_true=82.9851%

### partition: validation (n_rows=5,124)

**device_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 1 | 18043.7500 | 18286.5000 | 18480.2500 | 18620.0000 | 18705.0000 | 18715 | 17863.1073 |

**payment_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 1 | 17941.0000 | 18224.0000 | 18451.0000 | 18604.0000 | 18705.0000 | 18715 | 15470.8655 |

**merged_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 2 | 18061.0000 | 18292.5000 | 18487.0000 | 18626.0000 | 18705.7700 | 18715 | 18195.9873 |

**endpoints_same_component:** n_true=4,247 | n_false=877 | pct_true=82.8845%

### partition: embargo_2 (n_rows=2,711)

**device_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 1 | 18821.0000 | 18934.0000 | 19001.0000 | 19043.0000 | 19086.0000 | 19090 | 18442.5581 |

**payment_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 1 | 18791.0000 | 18914.0000 | 18996.0000 | 19042.0000 | 19086.0000 | 19090 | 16713.3622 |

**merged_component_size_total:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 2 | 18828.0000 | 18936.0000 | 19002.0000 | 19043.0000 | 19086.0000 | 19090 | 18867.8864 |

**endpoints_same_component:** n_true=2,335 | n_false=376 | pct_true=86.1306%


## 6. Fan-out-stratified diagnostic evaluation (isFraud, validation-partition only)

The ONE place `isFraud` is read in this phase -- strictly downstream of every quantity above, `validation`-partition rows only. Question: does `merged_component_size_total` / `endpoints_same_component` carry fraud-rate signal BEYOND device_to_payment's own prior fan-out (Phase E.1's M1 quantity), or is it just re-detecting fan-out/hub-domination (the same confound E.1's M5-vs-M5b correction addressed for raw overlap)? This is a one-time EDA evaluation only -- no production feature, score, threshold, config, or E.3 work is added regardless of the result below.

**Fixed bins, declared BEFORE `isFraud` was read** (not tuned to any fraud-rate outcome):

- `fanout_stratum_edges` (device_to_payment unbounded prior-distinct-partner count; E.1's own published p25/p50/p75 for this direction): `[274.0, 982.0, 2199.0]` -> ['low_fanout_lt_p25', 'mid_fanout_p25_to_p50', 'mid_fanout_p50_to_p75', 'high_fanout_ge_p75']
- `component_size_bin_edges` (`merged_component_size_total`; this run's own published overall p25/p50/p75/p90, section 4 above): `[7506.0, 12241.0, 15893.0, 17925.0]` -> ['merged_size_lt_p25', 'merged_size_p25_to_p50', 'merged_size_p50_to_p75', 'merged_size_p75_to_p90', 'merged_size_ge_p90']


`n_validation_rows` (partition == "validation" only): **5,124**


### 6.1 Unstratified (reference only)

n_rows=5,124 | fraud_rate=0.051717 | roc_auc_merged_component_size_total_vs_isFraud=0.5167

**fraud rate by merged_component_size_total bucket:**

| bucket | n_rows | fraud_rate |
|---|---|---|
| merged_size_lt_p25 | 24 | 0.166667 |
| merged_size_p25_to_p50 | 0 | nan |
| merged_size_p50_to_p75 | 0 | nan |
| merged_size_p75_to_p90 | 470 | 0.061702 |
| merged_size_ge_p90 | 4630 | 0.050108 |

**fraud rate by endpoints_same_component:**

| endpoints_same_component | n_rows | fraud_rate |
|---|---|---|
| True | 4247 | 0.051330 |
| False | 877 | 0.053592 |


### 6.2 Per fan-out stratum


**low_fanout_lt_p25** -- n_rows=1,436 | fraud_rate=0.052925 | roc_auc_merged_component_size_total_vs_isFraud=0.4735

fraud rate by merged_component_size_total bucket:

| bucket | n_rows | fraud_rate |
|---|---|---|
| merged_size_lt_p25 | 24 | 0.166667 |
| merged_size_p25_to_p50 | 0 | nan |
| merged_size_p50_to_p75 | 0 | nan |
| merged_size_p75_to_p90 | 113 | 0.070796 |
| merged_size_ge_p90 | 1299 | 0.049269 |

fraud rate by endpoints_same_component:

| endpoints_same_component | n_rows | fraud_rate |
|---|---|---|
| True | 1120 | 0.053571 |
| False | 316 | 0.050633 |


**mid_fanout_p25_to_p50** -- n_rows=906 | fraud_rate=0.046358 | roc_auc_merged_component_size_total_vs_isFraud=0.5492

fraud rate by merged_component_size_total bucket:

| bucket | n_rows | fraud_rate |
|---|---|---|
| merged_size_lt_p25 | 0 | nan |
| merged_size_p25_to_p50 | 0 | nan |
| merged_size_p50_to_p75 | 0 | nan |
| merged_size_p75_to_p90 | 137 | 0.021898 |
| merged_size_ge_p90 | 769 | 0.050715 |

fraud rate by endpoints_same_component:

| endpoints_same_component | n_rows | fraud_rate |
|---|---|---|
| True | 780 | 0.050000 |
| False | 126 | 0.023810 |


**mid_fanout_p50_to_p75** -- n_rows=777 | fraud_rate=0.030888 | roc_auc_merged_component_size_total_vs_isFraud=0.4775

fraud rate by merged_component_size_total bucket:

| bucket | n_rows | fraud_rate |
|---|---|---|
| merged_size_lt_p25 | 0 | nan |
| merged_size_p25_to_p50 | 0 | nan |
| merged_size_p50_to_p75 | 0 | nan |
| merged_size_p75_to_p90 | 45 | 0.088889 |
| merged_size_ge_p90 | 732 | 0.027322 |

fraud rate by endpoints_same_component:

| endpoints_same_component | n_rows | fraud_rate |
|---|---|---|
| True | 649 | 0.029276 |
| False | 128 | 0.039062 |


**high_fanout_ge_p75** -- n_rows=2,005 | fraud_rate=0.061347 | roc_auc_merged_component_size_total_vs_isFraud=0.5264

fraud rate by merged_component_size_total bucket:

| bucket | n_rows | fraud_rate |
|---|---|---|
| merged_size_lt_p25 | 0 | nan |
| merged_size_p25_to_p50 | 0 | nan |
| merged_size_p50_to_p75 | 0 | nan |
| merged_size_p75_to_p90 | 175 | 0.080000 |
| merged_size_ge_p90 | 1830 | 0.059563 |

fraud rate by endpoints_same_component:

| endpoints_same_component | n_rows | fraud_rate |
|---|---|---|
| True | 1698 | 0.058893 |
| False | 307 | 0.074919 |


### 6.3 Conclusion

ROC-AUC for merged_component_size_total vs. isFraud was computable in 4/4 fan-out strata; it was above 0.5 (better than chance ranking) in 2/4 of those. A monotonic non-decreasing fraud-rate gradient across the fixed merged_component_size_total buckets held in 1/4 strata. CONCLUSION: the pattern does not clearly persist once fan-out is controlled for -- consistent with the same population-generic-value/hub-domination confound Phase E.1's M5-vs-M5b correction found for raw overlap. Most of the unstratified signal, if any, is not shown to be beyond fan-out by this evaluation.

Per-stratum detail:
- low_fanout_lt_p25 (n=1436): AUC=0.4735, bucket fraud-rate monotonic non-decreasing=False
- mid_fanout_p25_to_p50 (n=906): AUC=0.5492, bucket fraud-rate monotonic non-decreasing=True
- mid_fanout_p50_to_p75 (n=777): AUC=0.4775, bucket fraud-rate monotonic non-decreasing=False
- high_fanout_ge_p75 (n=2005): AUC=0.5264, bucket fraud-rate monotonic non-decreasing=False


## 7. No production feature, scoring model, or E.3 work

Nothing in this report is a production feature, score, threshold, `configs/detection.yaml`-style config, or E.3 work -- this is a one-time EDA measurement + diagnostic evaluation only.

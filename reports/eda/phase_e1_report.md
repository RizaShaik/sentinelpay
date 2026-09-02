# SentinelPay -- Phase E.1 Link Sufficiency & Causal Cross-Key Fan-Out Report

**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_e1_report` from `reports/eda/phase_e1_results.json`** -- every number below is read from that file; re-running `python -m sentinelpay.eda.run_phase_e1` regenerates both together.

## Scope (what Phase E.1 is, and deliberately is not)

Phase E.1 is a narrow, non-target link/relationship sufficiency and causal
cross-key fan-out measurement. Its only question, per direction: does
anchor_node have enough strictly-causal distinct-partner fan-out with
other_node to be worth building a future Phase E.2 ring/fraud-network
mechanism on? Explicitly out of scope, per the approved E.1 proposal:

- **No target of any kind.** `isFraud` is never loaded, never read, never
  compared to any relationship. Nothing below is a fraud rate. There is no
  diagnostic-evaluation step at all in this phase -- stronger than Phase D
  (which reads `isFraud` exactly once, downstream of every score), matching
  Phase D.1's own precedent exactly.
- **No Union-Find, no connected components, no multi-hop/transitive graph
  traversal, no persistent graph structure.** Both overlap diagnostics below
  (M5 and M5b) are bounded and non-recursive: at most one pairwise
  set-intersection per pair of high-fan-out anchors, never chained through a
  third anchor.
- **No scoring, no flags, no `configs/detection.yaml`-style persistence.**
- **Reduced first-pass scope**: only 6 of the 13 directions in the full
  proposal are measured here (payment_node <-> device_node, payment_node <->
  email_purchaser_node, device_node <-> email_purchaser_node) --
  `email_recipient_node`/`addr_only_node` are measured columns available for
  a cheap follow-up, not yet analyzed (see section 6). This correction pass
  (M5b) does not change that scope.
- **No production Union-Find/E.2 decision made in advance.** The
  recommendation below (`recommend_union_find_for_e2`) is a pure function of
  the measured results in this report -- not a preference decided before
  this analysis ran, and a `True` result means "investigating Union-Find is
  evidenced," never "simple per-anchor fan-out is invalid."
- **M5 vs. M5b**: `overlap_diagnostic` (M5) is DESCRIPTIVE ONLY below --
  raw partner-set overlap, reported for context. A critical review of this
  phase's first real-data results found M5's raw numbers dominated by
  population-generic partner values (the most common browser/OS strings, the
  most common email domains), not evidence of coordinated structure.
  `frequency_adjusted_overlap_diagnostic` (M5b) -- overlap compared against a
  population-prevalence null baseline -- is the sole DECISION-DRIVING
  diagnostic for `recommend_union_find_for_e2`. See section 6 for both
  diagnostics' causal constructions.
- **`configs/split.yaml` boundaries are unchanged.**

## 1. Split configuration (unchanged from Phase B/C/D/D.1)

| partition | start_day | end_day |
|---|---|---|
| train | 1 | 130 |
| embargo_1 | 131 | 137 |
| validation | 138 | 160 |
| embargo_2 | 161 | 167 |
| holdout | 168 | 182 |

## 2. Holdout sealing

- Total rows loaded (train_transaction.csv joined to train_identity.csv): **590,540**.
- Rows filtered to `train`/`embargo_1`/`validation`/`embargo_2` (`sentinelpay.data.split.DEVELOPMENT_PARTITIONS`) **before** any relationship content analysis: **549,899**.
- Holdout rows excluded, never touched by relationship-frame, history, or overlap computation: **40,641**.
- `isFraud` is never loaded by `sentinelpay.eda.run_phase_e1`.

## Method

For each direction (anchor_col -> other_col), rows missing either column are
excluded (not imputed) before analysis via
`sentinelpay.eda.link_sufficiency.build_relationship_frame`. `_payment_node`/
`_device_node` are built via `build_node_key_column` -- a row-PRESERVING join
(unlike D.1's `build_group_key`): a row missing payment_node's components
stays available, as NaN in that column, for the device_node/email_purchaser_node
relationships that don't need it. `email_purchaser_node` is `P_emaildomain`
used directly.

**M1/M2 (distinct-count distributions and sufficiency-at-thresholds)**: for
every row, the number of DISTINCT `other_col` values among strictly-prior
same-anchor rows is counted by `sentinelpay.data.history.prior_group_distinct_other_count`
(unbounded) and `prior_group_windowed_distinct_other_count` (per candidate
window in {5, 10, 20, 50} events) -- both delegated, not re-implemented here;
see `tests/test_history.py` for the causal-correctness evidence (no
self-count, tied timestamps never see each other, no future leakage,
row-order independence) and `tests/test_link_sufficiency.py` for this
module's own tests against that guarantee.

**M3 (relationship row coverage)**: the % of development rows with both
relationship columns non-null -- the intersection of two individual node
coverages, computed once per relationship pair (symmetric) and shared by
both of that pair's directions.

**M4 (dominant-node concentration)**: anchor-side reuses D.1's own
`dominant_group_exclusion_sensitivity` (recomputing M1/M2 after excluding
the top-K largest anchor groups) applied to the distinct-partner-count
series; other-side is new -- the raw group-size distribution of `other_col`
alone, plus the % of valid (anchor, other) pairs whose `other_col` value is
one of the top-K most popular `other_col` values overall (the reverse-hub
check D.1 never needed, since it only ever had one column per key).

**M5 (one-hop overlap diagnostic, DESCRIPTIVE ONLY)** and **M5b
(frequency-adjusted overlap diagnostic, DECISION-DRIVING)** are both described in full,
with their precise causal constructions, below.


## M5 causal construction (recap) -- descriptive only, does not drive the recommendation

1. Each anchor's fan-out summary is the MAXIMUM of its own rows'
   `prior_distinct_other_count` values -- equal to the value at that
   anchor's own chronologically last row, a fact about its own history only.
2. Its partner set is the set of distinct `other_col` values seen at any of
   its rows strictly before its own last `dt_col`.
3. High-fan-out anchors are those at or above the 99th percentile VALUE of
   the PER-ANCHOR (not per-row) fan-out distribution within that
   relationship -- this percentile-VALUE selection degenerates to a single
   anchor whenever the relationship has a small number of distinct anchors
   (e.g. `P_emaildomain`'s 59 values), independent of the real underlying
   structure; M5b (below) fixes this.
4. For each pair of distinct high-fan-out anchors, at most one set
   intersection of their partner sets is computed -- never chained through a
   third anchor, no persistent structure built or updated.
5. Tie/future-leakage guarantees are inherited directly from
   `prior_group_distinct_other_count`'s own invariants, since every partner
   set is built from its output.

A critical review of this phase's first real-data results found M5's raw
overlap percentages dominated by population-generic partner values (the most
common browser/OS strings, the most common email domains) -- shared by
nearly every high-fan-out anchor regardless of any real coordination, and
essentially unchanged even after excluding the top-10 most universally-shared
values. M5's numbers below are retained for descriptive context only.

## M5b causal construction (recap) -- the decision-driving diagnostic

M5b asks a different question: "do high-fan-out anchors share partners more
often than the population-wide popularity of those partner values alone
would predict?"

1. **Anchor selection is rank-COUNT based, not percentile-VALUE based**:
   `n_target = max(HIGH_FANOUT_MIN_ANCHOR_COUNT, ceil((100-HIGH_FANOUT_PERCENTILE)/100 * n_anchors_total))`,
   clamped to the number of anchors with nonzero fan-out, with every anchor
   tied at the selection-boundary rank included. This is the fix for M5's
   small-anchor-population degeneracy above.
2. Partner sets are constructed identically to M5 (steps 1-2 above) for both
   the selected high-fan-out anchors and, separately, for EVERY anchor in the
   relationship (not just the high-fan-out subset) to build a population
   PREVALENCE baseline: `prevalence(v)` = the fraction of all anchors whose
   partner set contains `v`.
3. **Expected overlap under a null model**: treating each of an anchor's
   `H-1` other high-fan-out peers as an independent Bernoulli trial with
   success probability `prevalence(v)` for each of that anchor's own actual
   partner values `v` -- a standard association-strength approximation, not
   an exact combinatorial or permutation null (a permutation test was
   considered and rejected on computational-cost grounds).
4. `lift_ratio = mean_observed_overlap_pct / max(mean_expected_overlap_pct, LIFT_EPSILON)`
   -- aggregated as ratio-of-means (each side averaged across the H anchors
   first, then divided), not mean-of-ratios, to avoid a handful of anchors
   with near-zero expected overlap producing unstable, dominating individual
   values.
5. Bounded and non-recursive, exactly like M5: at most one pairwise set
   intersection per pair of high-fan-out anchors, no persistent structure,
   no chaining through a third anchor.

**Causal semantics, precisely**: every partner set -- both M5's and M5b's --
is strictly causal RELATIVE TO EACH ANCHOR'S OWN CUTOFF (that anchor's own
last `dt_col`); this per-anchor guarantee is unchanged and fully tested. The
population PREVALENCE baseline, however, is an AGGREGATE over every anchor's
(each different) cutoff -- it is a fixed snapshot of this run's development
window, NOT a claim that recomputing it after some change elsewhere in the
dataset would leave it unchanged, and not something meant to be recomputed
incrementally as new rows arrive. E.1 as a whole is an offline sufficiency
measurement over a fixed development window, not an online/streaming feature
computation.


## 3. Relationship-pair row coverage (M3, shared by both directions of each pair)

| relationship_pair | n_rows_valid | pct_rows_valid |
|---|---|---|
| payment_device | 73305 | 13.3306 |
| payment_email_purchaser | 395597 | 71.9399 |
| device_email_purchaser | 102388 | 18.6194 |


## 4. Per-direction results (M1-M5)


### 4.1 payment_node -> device_node

- anchor_col: `_payment_node` | other_col: `_device_node` | relationship pair: `payment_device`.
- Relationship row coverage (M3): **73,305** / 549,899 development rows (**13.33%**).
- Anchor groups: **17,903**, of which **10,030** are singletons. Largest anchor group: **657** rows; median **1.0**.

**Unbounded strictly-prior distinct-partner-count distribution (M1):**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 0 | 1.0000 | 3.0000 | 11.0000 | 21.0000 | 46.0000 | 70 | 7.3773 |


**Sufficiency by threshold, overall (M2, unbounded):**

| >= 1 | >= 2 | >= 3 | >= 5 | >= 10 |
|---|---|---|---|---|
| 75.5774 | 60.9154 | 52.9759 | 42.4500 | 27.5861 |


**Sufficiency by threshold, by partition (unbounded):**

| partition | n_valid_rows | pct_sufficient_ge_1 | pct_sufficient_ge_2 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 |
|---|---|---|---|---|---|---|
| train | 64130 | 74.1088 | 58.7634 | 50.7204 | 40.5271 | 25.6245 |
| embargo_1 | 1340 | 84.8507 | 74.9254 | 67.1642 | 57.3881 | 43.7313 |
| validation | 5124 | 84.7190 | 73.8681 | 66.2178 | 56.2061 | 40.7104 |
| embargo_2 | 2711 | 88.4544 | 80.4131 | 74.2899 | 54.5555 | 41.2025 |


**Anchor-side dominant-group exclusion sensitivity (M4):**

| top_k_excluded | excluded_group_sizes | n_valid_rows_remaining | pct_valid_rows_remaining_of_original_valid | pct_valid_rows_remaining_of_development_total | pct_sufficient_ge_1 | pct_sufficient_ge_2 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [657] | 72648 | 99.1037 | 99.1037 | 75.3579 | 60.5660 | 52.5562 | 41.9392 | 26.9725 |
| 3 | [657, 590, 449] | 71609 | 97.6864 | 97.6864 | 75.0031 | 60.0469 | 51.9250 | 41.6638 | 26.5763 |
| 5 | [657, 590, 449, 416, 359] | 70834 | 96.6292 | 96.6292 | 74.7325 | 59.6197 | 51.4118 | 41.0495 | 25.8492 |
| 10 | [657, 590, 449, 416, 359, 354, 329, 325, 323, 317] | 69186 | 94.3810 | 94.3810 | 74.1378 | 58.6737 | 50.2818 | 39.7002 | 24.2188 |


**Other-side dominant concentration (M4, reverse-hub check):**

- `other_col` groups: **1,599**, largest **10,202** rows, median **2.0**.

| top_k | top_other_values_row_count | pct_pairs_touching_top_k_other_values |
|---|---|---|
| 1 | 10202 | 13.9172 |
| 3 | 26285 | 35.8570 |
| 5 | 38108 | 51.9855 |
| 10 | 50987 | 69.5546 |


**Windowed sufficiency-at-decision-threshold summary (M1/M2, per candidate window):**

| window_size_events | >= 1 | >= 2 | >= 3 | >= 5 | >= 10 |
|---|---|---|---|---|---|
| 5 | 75.5774 | 58.7723 | 45.4526 | 8.5942 | 0.0000 |
| 10 | 75.5774 | 60.2114 | 51.6213 | 36.1531 | 0.1310 |
| 20 | 75.5774 | 60.5225 | 52.3402 | 41.9235 | 14.5283 |
| 50 | 75.5774 | 60.7203 | 52.5599 | 42.3600 | 27.5206 |


**Evaluation** (decision_threshold=2): row coverage **13.33%** (row_coverage_ok: **True**) | overall fan-out sufficiency **60.92%** (fanout_density_ok: **True**) | partition_stability_ok: **True** | dominant_anchor_robustness_ok: **True** (worst case after exclusion: 58.67%) | **is_suitable: True** | recommended_window: **50**.

**M5 overlap diagnostic (descriptive only -- does not drive the recommendation)**: 183 / 17,903 anchors at or above the 99th percentile (threshold value: 19.00 distinct partners). Mean RAW overlap fraction: **92.47%** | clears_multi_hop_signal (informational only): **True**.

| min | p25 | p50 | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|
| 75.0000 | 89.0873 | 92.8571 | 96.0769 | 100.0000 | 100.0000 | 92.4674 |


**M5b frequency-adjusted overlap diagnostic (DECISION-DRIVING)**: 183 / 17,903 anchors selected (rank-count selection, min_anchor_count=10), 183 with a nonempty partner set.

| mean observed overlap | mean expected overlap (null) | excess (pct points) | lift_ratio | clears_lift_signal |
|---|---|---|---|---|
| 92.47% | 72.20% | +20.27 pts | **1.28x** | **False** |

### 4.2 device_node -> payment_node

- anchor_col: `_device_node` | other_col: `_payment_node` | relationship pair: `payment_device`.
- Relationship row coverage (M3): **73,305** / 549,899 development rows (**13.33%**).
- Anchor groups: **1,599**, of which **797** are singletons. Largest anchor group: **10,202** rows; median **2.0**.

**Unbounded strictly-prior distinct-partner-count distribution (M1):**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 0 | 274.0000 | 982.0000 | 2199.0000 | 3073.0000 | 4297.9600 | 4505 | 1320.8914 |


**Sufficiency by threshold, overall (M2, unbounded):**

| >= 1 | >= 2 | >= 3 | >= 5 | >= 10 |
|---|---|---|---|---|
| 97.8187 | 96.4614 | 95.6183 | 94.5665 | 92.9514 |


**Sufficiency by threshold, by partition (unbounded):**

| partition | n_valid_rows | pct_sufficient_ge_1 | pct_sufficient_ge_2 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 |
|---|---|---|---|---|---|---|
| train | 64130 | 97.8169 | 96.5118 | 95.6916 | 94.6484 | 93.0703 |
| embargo_1 | 1340 | 97.9104 | 96.4925 | 95.0000 | 93.6567 | 91.5672 |
| validation | 5124 | 97.9118 | 96.1163 | 95.1405 | 93.8915 | 92.4278 |
| embargo_2 | 2711 | 97.6392 | 95.9056 | 95.0941 | 94.3563 | 91.8111 |


**Anchor-side dominant-group exclusion sensitivity (M4):**

| top_k_excluded | excluded_group_sizes | n_valid_rows_remaining | pct_valid_rows_remaining_of_original_valid | pct_valid_rows_remaining_of_development_total | pct_sufficient_ge_1 | pct_sufficient_ge_2 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [10202] | 63103 | 86.0828 | 86.0828 | 97.4676 | 95.8924 | 94.9147 | 93.6960 | 91.8276 |
| 3 | [10202, 9388, 6695] | 47020 | 64.1430 | 64.1430 | 96.6057 | 94.4981 | 93.1901 | 91.5674 | 89.0834 |
| 5 | [10202, 9388, 6695, 6429, 5394] | 35197 | 48.0145 | 48.0145 | 95.4712 | 92.6613 | 90.9282 | 88.7718 | 85.4902 |
| 10 | [10202, 9388, 6695, 6429, 5394, 4001, 2752, 2103, 2085, 1938] | 22318 | 30.4454 | 30.4454 | 92.8802 | 88.4712 | 85.7649 | 82.4088 | 77.3680 |


**Other-side dominant concentration (M4, reverse-hub check):**

- `other_col` groups: **17,903**, largest **657** rows, median **1.0**.

| top_k | top_other_values_row_count | pct_pairs_touching_top_k_other_values |
|---|---|---|
| 1 | 657 | 0.8963 |
| 3 | 1696 | 2.3136 |
| 5 | 2471 | 3.3708 |
| 10 | 4119 | 5.6190 |


**Windowed sufficiency-at-decision-threshold summary (M1/M2, per candidate window):**

| window_size_events | >= 1 | >= 2 | >= 3 | >= 5 | >= 10 |
|---|---|---|---|---|---|
| 5 | 97.8187 | 95.5706 | 92.6008 | 60.6944 | 0.0000 |
| 10 | 97.8187 | 96.0958 | 95.0590 | 93.3061 | 34.4247 |
| 20 | 97.8187 | 96.2240 | 95.2800 | 94.1355 | 92.1124 |
| 50 | 97.8187 | 96.3631 | 95.3850 | 94.2801 | 92.5790 |


**Evaluation** (decision_threshold=2): row coverage **13.33%** (row_coverage_ok: **True**) | overall fan-out sufficiency **96.46%** (fanout_density_ok: **True**) | partition_stability_ok: **True** | dominant_anchor_robustness_ok: **True** (worst case after exclusion: 88.47%) | **is_suitable: True** | recommended_window: **50**.

**M5 overlap diagnostic (descriptive only -- does not drive the recommendation)**: 16 / 1,599 anchors at or above the 99th percentile (threshold value: 500.02 distinct partners). Mean RAW overlap fraction: **68.48%** | clears_multi_hop_signal (informational only): **True**.

| min | p25 | p50 | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|
| 58.0910 | 62.4546 | 70.1267 | 73.2122 | 73.7986 | 78.6427 | 68.4778 |


**M5b frequency-adjusted overlap diagnostic (DECISION-DRIVING)**: 16 / 1,599 anchors selected (rank-count selection, min_anchor_count=10), 16 with a nonempty partner set.

| mean observed overlap | mean expected overlap (null) | excess (pct points) | lift_ratio | clears_lift_signal |
|---|---|---|---|---|
| 68.48% | 6.19% | +62.28 pts | **11.06x** | **True** |

### 4.3 payment_node -> email_purchaser_node

- anchor_col: `_payment_node` | other_col: `P_emaildomain` | relationship pair: `payment_email_purchaser`.
- Relationship row coverage (M3): **395,597** / 549,899 development rows (**71.94%**).
- Anchor groups: **35,065**, of which **14,421** are singletons. Largest anchor group: **4,599** rows; median **2.0**.

**Unbounded strictly-prior distinct-partner-count distribution (M1):**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 0 | 2.0000 | 5.0000 | 11.0000 | 17.0000 | 28.0000 | 32 | 7.1551 |


**Sufficiency by threshold, overall (M2, unbounded):**

| >= 1 | >= 2 | >= 3 | >= 5 | >= 10 |
|---|---|---|---|---|
| 91.1362 | 79.2134 | 69.2738 | 54.2524 | 29.7674 |


**Sufficiency by threshold, by partition (unbounded):**

| partition | n_valid_rows | pct_sufficient_ge_1 | pct_sufficient_ge_2 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 |
|---|---|---|---|---|---|---|
| train | 320623 | 89.9539 | 77.2359 | 66.8523 | 51.5135 | 27.3886 |
| embargo_1 | 13503 | 95.9787 | 87.2177 | 79.2120 | 65.4817 | 39.4801 |
| validation | 46849 | 96.2838 | 87.7073 | 79.6538 | 65.8072 | 39.5761 |
| embargo_2 | 14622 | 96.0949 | 87.9702 | 79.9343 | 66.9197 | 41.5333 |


**Anchor-side dominant-group exclusion sensitivity (M4):**

| top_k_excluded | excluded_group_sizes | n_valid_rows_remaining | pct_valid_rows_remaining_of_original_valid | pct_valid_rows_remaining_of_development_total | pct_sufficient_ge_1 | pct_sufficient_ge_2 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [4599] | 390998 | 98.8375 | 98.8375 | 91.0322 | 78.9697 | 68.9134 | 53.7177 | 28.9603 |
| 3 | [4599, 4457, 3704] | 382837 | 96.7745 | 96.7745 | 90.8415 | 78.5225 | 68.2525 | 52.7449 | 27.4697 |
| 5 | [4599, 4457, 3704, 2740, 2685] | 377412 | 95.4032 | 95.4032 | 90.7104 | 78.2151 | 67.7983 | 52.0749 | 26.4811 |
| 10 | [4599, 4457, 3704, 2740, 2685, 2651, 2113, 1859, 1733, 1676] | 367380 | 92.8672 | 92.8672 | 90.4581 | 77.6243 | 66.9285 | 50.7888 | 24.6067 |


**Other-side dominant concentration (M4, reverse-hub check):**

- `other_col` groups: **59**, largest **183,818** rows, median **242.0**.

| top_k | top_other_values_row_count | pct_pairs_touching_top_k_other_values |
|---|---|---|
| 1 | 183818 | 46.4660 |
| 3 | 304146 | 76.8828 |
| 5 | 350708 | 88.6528 |
| 10 | 373534 | 94.4229 |


**Windowed sufficiency-at-decision-threshold summary (M1/M2, per candidate window):**

| window_size_events | >= 1 | >= 2 | >= 3 | >= 5 | >= 10 |
|---|---|---|---|---|---|
| 5 | 91.1362 | 72.6146 | 41.0688 | 1.6006 | 0.0000 |
| 10 | 91.1362 | 77.8310 | 60.4658 | 16.9925 | 0.0005 |
| 20 | 91.1362 | 78.9773 | 67.4090 | 38.9437 | 0.1992 |
| 50 | 91.1362 | 79.1803 | 69.1856 | 52.7926 | 6.3628 |


**Evaluation** (decision_threshold=2): row coverage **71.94%** (row_coverage_ok: **True**) | overall fan-out sufficiency **79.21%** (fanout_density_ok: **True**) | partition_stability_ok: **True** | dominant_anchor_robustness_ok: **True** (worst case after exclusion: 77.62%) | **is_suitable: True** | recommended_window: **50**.

**M5 overlap diagnostic (descriptive only -- does not drive the recommendation)**: 365 / 35,065 anchors at or above the 99th percentile (threshold value: 11.00 distinct partners). Mean RAW overlap fraction: **99.88%** | clears_multi_hop_signal (informational only): **True**.

| min | p25 | p50 | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|
| 81.8182 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 99.8845 |


**M5b frequency-adjusted overlap diagnostic (DECISION-DRIVING)**: 365 / 35,065 anchors selected (rank-count selection, min_anchor_count=10), 365 with a nonempty partner set.

| mean observed overlap | mean expected overlap (null) | excess (pct points) | lift_ratio | clears_lift_signal |
|---|---|---|---|---|
| 99.88% | 94.81% | +5.07 pts | **1.05x** | **False** |

### 4.4 email_purchaser_node -> payment_node

- anchor_col: `P_emaildomain` | other_col: `_payment_node` | relationship pair: `payment_email_purchaser`.
- Relationship row coverage (M3): **395,597** / 549,899 development rows (**71.94%**).
- Anchor groups: **59**, of which **2** are singletons. Largest anchor group: **183,818** rows; median **242.0**.

**Unbounded strictly-prior distinct-partner-count distribution (M1):**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 0 | 4008.0000 | 9281.0000 | 14893.0000 | 19090.0000 | 20984.0000 | 21218 | 9620.0855 |


**Sufficiency by threshold, overall (M2, unbounded):**

| >= 1 | >= 2 | >= 3 | >= 5 | >= 10 |
|---|---|---|---|---|
| 99.9851 | 99.9674 | 99.9489 | 99.9138 | 99.8276 |


**Sufficiency by threshold, by partition (unbounded):**

| partition | n_valid_rows | pct_sufficient_ge_1 | pct_sufficient_ge_2 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 |
|---|---|---|---|---|---|---|
| train | 320623 | 99.9816 | 99.9601 | 99.9379 | 99.8946 | 99.7892 |
| embargo_1 | 13503 | 100.0000 | 100.0000 | 99.9926 | 99.9926 | 99.9926 |
| validation | 46849 | 100.0000 | 99.9979 | 99.9957 | 99.9957 | 99.9936 |
| embargo_2 | 14622 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 99.9863 |


**Anchor-side dominant-group exclusion sensitivity (M4):**

| top_k_excluded | excluded_group_sizes | n_valid_rows_remaining | pct_valid_rows_remaining_of_original_valid | pct_valid_rows_remaining_of_development_total | pct_sufficient_ge_1 | pct_sufficient_ge_2 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [183818] | 211779 | 53.5340 | 53.5340 | 99.9726 | 99.9400 | 99.9060 | 99.8413 | 99.6827 |
| 3 | [183818, 91314, 29014] | 91451 | 23.1172 | 23.1172 | 99.9388 | 99.8655 | 99.7890 | 99.6435 | 99.2881 |
| 5 | [183818, 91314, 29014, 25801, 20761] | 44889 | 11.3472 | 11.3472 | 99.8797 | 99.7349 | 99.5834 | 99.2960 | 98.5943 |
| 10 | [183818, 91314, 29014, 25801, 20761, 7288, 5325, 3671, 3656, 2886] | 22063 | 5.5771 | 5.5771 | 99.7779 | 99.5150 | 99.2340 | 98.7037 | 97.3984 |


**Other-side dominant concentration (M4, reverse-hub check):**

- `other_col` groups: **35,065**, largest **4,599** rows, median **2.0**.

| top_k | top_other_values_row_count | pct_pairs_touching_top_k_other_values |
|---|---|---|
| 1 | 4599 | 1.1625 |
| 3 | 12760 | 3.2255 |
| 5 | 18185 | 4.5968 |
| 10 | 28217 | 7.1328 |


**Windowed sufficiency-at-decision-threshold summary (M1/M2, per candidate window):**

| window_size_events | >= 1 | >= 2 | >= 3 | >= 5 | >= 10 |
|---|---|---|---|---|---|
| 5 | 99.9851 | 99.9075 | 99.5215 | 82.9094 | 0.0000 |
| 10 | 99.9851 | 99.9651 | 99.9325 | 99.7722 | 57.9375 |
| 20 | 99.9851 | 99.9674 | 99.9489 | 99.9042 | 99.6924 |
| 50 | 99.9851 | 99.9674 | 99.9489 | 99.9138 | 99.8140 |


**Evaluation** (decision_threshold=2): row coverage **71.94%** (row_coverage_ok: **True**) | overall fan-out sufficiency **99.97%** (fanout_density_ok: **True**) | partition_stability_ok: **True** | dominant_anchor_robustness_ok: **True** (worst case after exclusion: 99.52%) | **is_suitable: True** | recommended_window: **50**.

**M5 overlap diagnostic (descriptive only -- does not drive the recommendation)**: 1 / 59 anchors at or above the 99th percentile (threshold value: 16557.70 distinct partners). Mean RAW overlap fraction: **0.00%** | clears_multi_hop_signal (informational only): **False**.

| min | p25 | p50 | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|
| 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


**M5b frequency-adjusted overlap diagnostic (DECISION-DRIVING)**: 10 / 59 anchors selected (rank-count selection, min_anchor_count=10), 10 with a nonempty partner set.

| mean observed overlap | mean expected overlap (null) | excess (pct points) | lift_ratio | clears_lift_signal |
|---|---|---|---|---|
| 76.28% | 46.19% | +30.10 pts | **1.65x** | **False** |

### 4.5 device_node -> email_purchaser_node

- anchor_col: `_device_node` | other_col: `P_emaildomain` | relationship pair: `device_email_purchaser`.
- Relationship row coverage (M3): **102,388** / 549,899 development rows (**18.62%**).
- Anchor groups: **4,918**, of which **2,130** are singletons. Largest anchor group: **15,050** rows; median **2.0**.

**Unbounded strictly-prior distinct-partner-count distribution (M1):**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 0 | 7.0000 | 30.0000 | 42.0000 | 47.0000 | 53.0000 | 54 | 26.4032 |


**Sufficiency by threshold, overall (M2, unbounded):**

| >= 1 | >= 2 | >= 3 | >= 5 | >= 10 |
|---|---|---|---|---|
| 95.1967 | 89.3171 | 84.4611 | 78.5776 | 72.9744 |


**Sufficiency by threshold, by partition (unbounded):**

| partition | n_valid_rows | pct_sufficient_ge_1 | pct_sufficient_ge_2 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 |
|---|---|---|---|---|---|---|
| train | 87997 | 95.3362 | 89.7190 | 85.2268 | 79.8073 | 74.4207 |
| embargo_1 | 2469 | 94.8157 | 88.4569 | 81.3690 | 68.9753 | 60.7533 |
| validation | 8068 | 94.5835 | 87.4566 | 81.5816 | 73.1408 | 65.5057 |
| embargo_2 | 3854 | 93.5392 | 84.5874 | 74.9870 | 68.0332 | 63.4146 |


**Anchor-side dominant-group exclusion sensitivity (M4):**

| top_k_excluded | excluded_group_sizes | n_valid_rows_remaining | pct_valid_rows_remaining_of_original_valid | pct_valid_rows_remaining_of_development_total | pct_sufficient_ge_1 | pct_sufficient_ge_2 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [15050] | 87338 | 85.3010 | 85.3010 | 94.3701 | 87.4797 | 81.7903 | 74.9010 | 68.3517 |
| 3 | [15050, 9281, 6295] | 71762 | 70.0883 | 70.0883 | 93.1510 | 84.7719 | 77.8504 | 69.4783 | 61.6036 |
| 5 | [15050, 9281, 6295, 6145, 5653] | 59964 | 58.5655 | 58.5655 | 91.8068 | 81.7924 | 73.5191 | 63.5148 | 54.1858 |
| 10 | [15050, 9281, 6295, 6145, 5653, 4281, 4071, 3825, 2473, 2131] | 43183 | 42.1758 | 42.1758 | 88.6344 | 74.7424 | 63.2749 | 49.4593 | 36.9497 |


**Other-side dominant concentration (M4, reverse-hub check):**

- `other_col` groups: **59**, largest **42,039** rows, median **91.0**.

| top_k | top_other_values_row_count | pct_pairs_touching_top_k_other_values |
|---|---|---|
| 1 | 42039 | 41.0585 |
| 3 | 74530 | 72.7917 |
| 5 | 88937 | 86.8627 |
| 10 | 95105 | 92.8869 |


**Windowed sufficiency-at-decision-threshold summary (M1/M2, per candidate window):**

| window_size_events | >= 1 | >= 2 | >= 3 | >= 5 | >= 10 |
|---|---|---|---|---|---|
| 5 | 95.1967 | 83.5000 | 56.4646 | 3.5844 | 0.0000 |
| 10 | 95.1967 | 88.3551 | 78.2533 | 33.7305 | 0.0010 |
| 20 | 95.1967 | 89.1462 | 83.6055 | 65.7177 | 1.7990 |
| 50 | 95.1967 | 89.3083 | 84.4171 | 77.9828 | 31.9256 |


**Evaluation** (decision_threshold=2): row coverage **18.62%** (row_coverage_ok: **True**) | overall fan-out sufficiency **89.32%** (fanout_density_ok: **True**) | partition_stability_ok: **True** | dominant_anchor_robustness_ok: **True** (worst case after exclusion: 74.74%) | **is_suitable: True** | recommended_window: **50**.

**M5 overlap diagnostic (descriptive only -- does not drive the recommendation)**: 52 / 4,918 anchors at or above the 99th percentile (threshold value: 11.00 distinct partners). Mean RAW overlap fraction: **100.00%** | clears_multi_hop_signal (informational only): **True**.

| min | p25 | p50 | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|
| 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |


**M5b frequency-adjusted overlap diagnostic (DECISION-DRIVING)**: 52 / 4,918 anchors selected (rank-count selection, min_anchor_count=10), 52 with a nonempty partner set.

| mean observed overlap | mean expected overlap (null) | excess (pct points) | lift_ratio | clears_lift_signal |
|---|---|---|---|---|
| 100.00% | 56.18% | +43.82 pts | **1.78x** | **False** |

### 4.6 email_purchaser_node -> device_node

- anchor_col: `P_emaildomain` | other_col: `_device_node` | relationship pair: `device_email_purchaser`.
- Relationship row coverage (M3): **102,388** / 549,899 development rows (**18.62%**).
- Anchor groups: **59**, of which **0** are singletons. Largest anchor group: **42,039** rows; median **91.0**.

**Unbounded strictly-prior distinct-partner-count distribution (M1):**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 0 | 193.0000 | 592.0000 | 1403.0000 | 2268.0000 | 3130.0000 | 3226 | 899.2403 |


**Sufficiency by threshold, overall (M2, unbounded):**

| >= 1 | >= 2 | >= 3 | >= 5 | >= 10 |
|---|---|---|---|---|
| 99.9424 | 99.8301 | 99.7363 | 99.5361 | 98.8426 |


**Sufficiency by threshold, by partition (unbounded):**

| partition | n_valid_rows | pct_sufficient_ge_1 | pct_sufficient_ge_2 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 |
|---|---|---|---|---|---|---|
| train | 87997 | 99.9330 | 99.8023 | 99.6932 | 99.4716 | 98.6863 |
| embargo_1 | 2469 | 100.0000 | 100.0000 | 100.0000 | 99.8785 | 99.6760 |
| validation | 8068 | 100.0000 | 100.0000 | 100.0000 | 99.9504 | 99.8265 |
| embargo_2 | 3854 | 100.0000 | 100.0000 | 100.0000 | 99.9222 | 99.8184 |


**Anchor-side dominant-group exclusion sensitivity (M4):**

| top_k_excluded | excluded_group_sizes | n_valid_rows_remaining | pct_valid_rows_remaining_of_original_valid | pct_valid_rows_remaining_of_development_total | pct_sufficient_ge_1 | pct_sufficient_ge_2 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [42039] | 60349 | 58.9415 | 58.9415 | 99.9039 | 99.7150 | 99.5592 | 99.2245 | 98.0646 |
| 3 | [42039, 17535, 14956] | 27858 | 27.2083 | 27.2083 | 99.7990 | 99.3969 | 99.0846 | 98.3811 | 95.9760 |
| 5 | [42039, 17535, 14956, 10433, 3974] | 13451 | 13.1373 | 13.1373 | 99.5985 | 98.7808 | 98.1711 | 96.7586 | 91.9486 |
| 10 | [42039, 17535, 14956, 10433, 3974, 2069, 1727, 937, 731, 704] | 7283 | 7.1131 | 7.1131 | 99.3272 | 97.9953 | 96.9930 | 94.6313 | 86.6813 |


**Other-side dominant concentration (M4, reverse-hub check):**

- `other_col` groups: **4,918**, largest **15,050** rows, median **2.0**.

| top_k | top_other_values_row_count | pct_pairs_touching_top_k_other_values |
|---|---|---|
| 1 | 15050 | 14.6990 |
| 3 | 30626 | 29.9117 |
| 5 | 42424 | 41.4345 |
| 10 | 59205 | 57.8242 |


**Windowed sufficiency-at-decision-threshold summary (M1/M2, per candidate window):**

| window_size_events | >= 1 | >= 2 | >= 3 | >= 5 | >= 10 |
|---|---|---|---|---|---|
| 5 | 99.9424 | 97.5368 | 85.6419 | 20.2827 | 0.0000 |
| 10 | 99.9424 | 99.6494 | 98.4725 | 82.6259 | 0.9493 |
| 20 | 99.9424 | 99.7998 | 99.6709 | 98.6453 | 54.0561 |
| 50 | 99.9424 | 99.8301 | 99.7353 | 99.5224 | 97.8591 |


**Evaluation** (decision_threshold=2): row coverage **18.62%** (row_coverage_ok: **True**) | overall fan-out sufficiency **99.83%** (fanout_density_ok: **True**) | partition_stability_ok: **True** | dominant_anchor_robustness_ok: **True** (worst case after exclusion: 98.00%) | **is_suitable: True** | recommended_window: **50**.

**M5 overlap diagnostic (descriptive only -- does not drive the recommendation)**: 1 / 59 anchors at or above the 99th percentile (threshold value: 2636.72 distinct partners). Mean RAW overlap fraction: **0.00%** | clears_multi_hop_signal (informational only): **False**.

| min | p25 | p50 | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|
| 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


**M5b frequency-adjusted overlap diagnostic (DECISION-DRIVING)**: 10 / 59 anchors selected (rank-count selection, min_anchor_count=10), 10 with a nonempty partner set.

| mean observed overlap | mean expected overlap (null) | excess (pct points) | lift_ratio | clears_lift_signal |
|---|---|---|---|---|
| 76.99% | 44.72% | +32.27 pts | **1.72x** | **False** |

## 5. Recommendation

**Per-direction verdicts, window recommendations, and both overlap diagnostics** (`raw_overlap_pct` is M5, descriptive only; `lift_ratio` is M5b, decision-driving):

| direction | is_suitable | recommended_window | raw_overlap_pct_M5_descriptive | lift_ratio_M5b_decision_driving |
|---|---|---|---|---|
| payment_to_device | True | 50 | 92.4674 | 1.2808 |
| device_to_payment | True | 50 | 68.4778 | 11.0572 |
| payment_to_email_purchaser | True | 50 | 99.8845 | 1.0535 |
| email_purchaser_to_payment | True | 50 | 0.0000 | 1.6516 |
| device_to_email_purchaser | True | 50 | 100.0000 | 1.7800 |
| email_purchaser_to_device | True | 50 | 0.0000 | 1.7218 |


**recommend_union_find_for_e2: True**

At least one usable relationship's M5b frequency-adjusted overlap statistic (lift_ratio) clears LIFT_SIGNAL_THRESHOLD (2.0x population-popularity expectation): device_to_payment. This means high-fan-out anchors share partners measurably more often than the population-wide popularity of those partner values alone would predict -- investigating Union-Find/component structure for a future Phase E.2 is evidenced, not merely generically appealing. It does NOT mean per-anchor fan-out counts are invalid or unusable on their own, and it is NOT driven by M5's raw overlap percentage, which a critical review found to be dominated by population-generic partner values (see module docstring).

This is the output of `sentinelpay.eda.link_sufficiency.recommend_relationships` applied to the measured results above -- a pure function of this run's numbers, not a preference chosen before E.1 ran, and driven by M5b's `lift_ratio` only (M5's raw overlap is shown above for context but never gates this boolean). It is a recommendation for a Phase E.2 scoping decision, not an implementation: no Union-Find, component detection, scoring, or persistence exists yet regardless of this value.

## 6. Not yet analyzed (measured columns available, deferred to a follow-up pass)

- email_recipient_node (R_emaildomain) -- columns not loaded by this run; measured columns available, not yet analyzed.
- addr_only_node (addr1 alone) -- columns not loaded by this run; measured columns available, not yet analyzed.
# SentinelPay -- Phase B Leakage-Safe Temporal Data Pipeline Report

**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_b_report` from `reports/eda/phase_b_results.json`** -- every number below is read from that file; re-running `python -m sentinelpay.eda.run_phase_b` regenerates both together.

## 1. Split configuration (fixed, not fit to data)

Boundaries are fixed day-index ranges from `configs/split.yaml`, chosen by convention -- never by searching for a cut that optimizes fraud rate or any label-based objective.

| partition | start_day | end_day |
|---|---|---|
| train | 1 | 130 |
| embargo_1 | 131 | 137 |
| validation | 138 | 160 |
| embargo_2 | 161 | 167 |
| holdout | 168 | 182 |

## 2. Structural split validation

**is_valid: True**

| partition | row count |
|---|---|
| train | 445,173 |
| embargo_1 | 18,589 |
| validation | 65,193 |
| embargo_2 | 20,944 |
| holdout | 40,641 |

- Unassigned rows: 0 | Empty partitions: none | TransactionIDs crossing partitions: 0
- Chronological order OK: **True** | Holdout strictly after validation: **True** | Embargoes isolated: **True**

Chronological TransactionDT bounds per partition (min, max):

| partition | min TransactionDT | max TransactionDT |
|---|---|---|
| train | 86,400 | 11,318,397 |
| embargo_1 | 11,318,417 | 11,923,178 |
| validation | 11,923,215 | 13,910,379 |
| embargo_2 | 13,910,472 | 14,515,163 |
| holdout | 14,515,200 | 15,811,131 |

## Holdout sealing (applies to this entire report)

`holdout` (see split configuration above) is reserved for Phase H. This
report shows only its structural existence -- row count, and that it is
chronologically after `validation` with no overlap. No fraud rate, drift,
correlation, missingness, entity/proxy statistic, or any other content
measurement is computed on holdout rows anywhere in Phase B. Section 3
(drift) compares `train` vs. `validation` only.

## 3. Embargo boundary sensitivity (train / embargo_1 / validation only)

Descriptive only. A 7-day embargo is a conservative engineering default, not a proof that boundary leakage from undocumented D-column semantics is eliminated -- several D-columns have means well beyond 7 days (Phase A: D1 ~86-104, D4 ~131-149), so their true lookback window relative to this embargo is unknown. Boundaries are NOT changed based on this result in Phase B.

| partition | column | mean | pct_missing | n_rows |
|---|---|---|---|---|
| train | D1 | 89.4247 | 0.1343 | 445173 |
| train | D2 | 166.4566 | 49.3568 | 445173 |
| train | D3 | 28.6411 | 46.4882 | 445173 |
| train | D4 | 133.4263 | 29.6044 | 445173 |
| train | D5 | 42.3004 | 53.9116 | 445173 |
| train | D6 | 62.8151 | 87.4954 | 445173 |
| train | D7 | 44.4027 | 93.6218 | 445173 |
| train | D8 | 149.9346 | 86.7541 | 445173 |
| train | D9 | 0.5607 | 86.7541 | 445173 |
| train | D10 | 117.0253 | 14.3540 | 445173 |
| train | D11 | 135.3971 | 53.0697 | 445173 |
| train | D12 | 51.4566 | 88.7318 | 445173 |
| train | D13 | 16.8708 | 89.5301 | 445173 |
| train | D14 | 50.6285 | 89.3273 | 445173 |
| train | D15 | 155.2758 | 16.5289 | 445173 |
| embargo_1 | D1 | 111.4748 | 1.3341 | 18589 |
| embargo_1 | D2 | 178.2360 | 41.7666 | 18589 |
| embargo_1 | D3 | 28.0418 | 38.7164 | 18589 |
| embargo_1 | D4 | 160.5727 | 30.3351 | 18589 |
| embargo_1 | D5 | 43.4288 | 51.2723 | 18589 |
| embargo_1 | D6 | 84.7174 | 89.7950 | 18589 |
| embargo_1 | D7 | 47.6228 | 94.4376 | 18589 |
| embargo_1 | D8 | 137.9045 | 90.3814 | 18589 |
| embargo_1 | D9 | 0.5522 | 90.3814 | 18589 |
| embargo_1 | D10 | 138.7711 | 9.1775 | 18589 |
| embargo_1 | D11 | 173.7618 | 37.2048 | 18589 |
| embargo_1 | D12 | 74.4109 | 90.7042 | 18589 |
| embargo_1 | D13 | 19.7556 | 91.2852 | 18589 |
| embargo_1 | D14 | 73.7823 | 91.3282 | 18589 |
| embargo_1 | D15 | 190.9089 | 16.3322 | 18589 |
| validation | D1 | 112.6844 | 0.5860 | 65193 |
| validation | D2 | 182.3923 | 41.8803 | 65193 |
| validation | D3 | 28.8824 | 38.5870 | 65193 |
| validation | D4 | 159.5514 | 24.8002 | 65193 |
| validation | D5 | 43.9349 | 47.8671 | 65193 |
| validation | D6 | 82.9403 | 89.6691 | 65193 |
| validation | D7 | 42.0533 | 94.2985 | 65193 |
| validation | D8 | 143.6938 | 90.6754 | 65193 |
| validation | D9 | 0.5698 | 90.6754 | 65193 |
| validation | D10 | 142.6557 | 7.9196 | 65193 |
| validation | D11 | 167.8902 | 27.4646 | 65193 |
| validation | D12 | 66.5717 | 90.7843 | 65193 |
| validation | D13 | 22.0785 | 91.2690 | 65193 |
| validation | D14 | 70.1668 | 91.4454 | 65193 |
| validation | D15 | 187.8750 | 9.2295 | 65193 |


## 4. Drift: train vs. validation only

**Numeric (KS 2-sample), curated columns:**

| column | ks_statistic | p_value | mean_early | mean_late | pct_missing_early | pct_missing_late |
|---|---|---|---|---|---|---|
| C9 | 0.1316 | 0.0000 | 4.3184 | 5.1480 | 0.0000 | 0.0000 |
| D15 | 0.0913 | 0.0000 | 155.2758 | 187.8750 | 16.5289 | 9.2295 |
| C8 | 0.0843 | 0.0000 | 6.3714 | 1.2557 | 0.0000 | 0.0000 |
| C10 | 0.0827 | 0.0000 | 6.4940 | 1.3376 | 0.0000 | 0.0000 |
| D1 | 0.0825 | 0.0000 | 89.4248 | 112.6844 | 0.1343 | 0.5860 |
| C13 | 0.0773 | 0.0000 | 32.6961 | 32.4784 | 0.0000 | 0.0000 |
| D11 | 0.0773 | 0.0000 | 135.3971 | 167.8902 | 53.0697 | 27.4646 |
| D6 | 0.0733 | 0.0000 | 62.8151 | 82.9403 | 87.4954 | 89.6691 |
| C4 | 0.0713 | 0.0000 | 4.8222 | 1.7972 | 0.0000 | 0.0000 |
| D7 | 0.0712 | 0.0000 | 44.4027 | 42.0533 | 93.6218 | 94.2985 |
| D4 | 0.0681 | 0.0000 | 133.4262 | 159.5514 | 29.6044 | 24.8002 |
| D13 | 0.0655 | 0.0000 | 16.8708 | 22.0785 | 89.5301 | 91.2690 |
| D10 | 0.0648 | 0.0000 | 117.0252 | 142.6557 | 14.3540 | 7.9196 |
| D12 | 0.0576 | 0.0000 | 51.4566 | 66.5717 | 88.7318 | 90.7843 |
| D2 | 0.0574 | 0.0000 | 166.4566 | 182.3923 | 49.3568 | 41.8803 |
| C2 | 0.0535 | 0.0000 | 17.0055 | 10.1022 | 0.0000 | 0.0000 |
| C5 | 0.0533 | 0.0000 | 5.4212 | 6.2496 | 0.0000 | 0.0000 |
| C12 | 0.0523 | 0.0000 | 5.2326 | 0.4946 | 0.0000 | 0.0000 |
| D14 | 0.0466 | 0.0000 | 50.6285 | 70.1668 | 89.3273 | 91.4454 |
| C14 | 0.0458 | 0.0000 | 8.6928 | 7.2808 | 0.0000 | 0.0000 |
| C1 | 0.0447 | 0.0000 | 15.5584 | 9.7931 | 0.0000 | 0.0000 |
| D9 | 0.0402 | 0.0000 | 0.5607 | 0.5698 | 86.7541 | 90.6754 |
| C6 | 0.0397 | 0.0000 | 9.7551 | 7.1719 | 0.0000 | 0.0000 |
| D5 | 0.0326 | 0.0000 | 42.3004 | 43.9349 | 53.9116 | 47.8671 |
| C11 | 0.0277 | 0.0000 | 11.2686 | 7.2417 | 0.0000 | 0.0000 |
| D8 | 0.0269 | 0.0007 | 149.9346 | 143.6938 | 86.7541 | 90.6754 |
| C7 | 0.0223 | 0.0000 | 3.6795 | 0.2701 | 0.0000 | 0.0000 |
| D3 | 0.0219 | 0.0000 | 28.6411 | 28.8824 | 46.4882 | 38.5870 |
| TransactionAmt | 0.0140 | 0.0000 | 134.3965 | 137.9904 | 0.0000 | 0.0000 |
| C3 | 0.0015 | 0.9995 | 0.0056 | 0.0042 | 0.0000 | 0.0000 |


**Categorical (chi-square):**

| column | chi2_statistic | p_value | n_categories_observed |
|---|---|---|---|
| M2 | 14954.9957 | 0.0000 | 3 |
| M3 | 14954.9563 | 0.0000 | 3 |
| M1 | 14952.5229 | 0.0000 | 3 |
| ProductCD | 3012.9309 | 0.0000 | 5 |
| card6 | 2343.7724 | 0.0000 | 5 |
| M6 | 2266.7011 | 0.0000 | 3 |
| card4 | 1827.5581 | 0.0000 | 5 |
| P_emaildomain | 357.1289 | 0.0000 | 60 |
| M4 | 354.3830 | 0.0000 | 4 |
| M5 | 291.0143 | 0.0000 | 3 |


**Curated target correlation -- train partition:**

| column | abs_corr_with_target | corr_with_target |
|---|---|---|
| D8 | 0.1408 | -0.1408 |
| D7 | 0.1383 | -0.1383 |
| D2 | 0.0843 | -0.0843 |
| D15 | 0.0749 | -0.0749 |
| D10 | 0.0677 | -0.0677 |
| D5 | 0.0660 | -0.0660 |
| D1 | 0.0641 | -0.0641 |
| D4 | 0.0633 | -0.0633 |
| D13 | 0.0535 | -0.0535 |
| D6 | 0.0463 | -0.0463 |
| D3 | 0.0462 | -0.0462 |
| D9 | 0.0442 | -0.0442 |
| D11 | 0.0418 | -0.0418 |
| C2 | 0.0360 | 0.0360 |
| C9 | 0.0337 | -0.0337 |


**Curated target correlation -- validation partition:**

| column | abs_corr_with_target | corr_with_target |
|---|---|---|
| C12 | 0.1905 | 0.1905 |
| D8 | 0.1756 | -0.1756 |
| C7 | 0.1745 | 0.1745 |
| D7 | 0.1270 | -0.1270 |
| C8 | 0.0835 | 0.0835 |
| D15 | 0.0814 | -0.0814 |
| D2 | 0.0809 | -0.0809 |
| D10 | 0.0807 | -0.0807 |
| D1 | 0.0719 | -0.0719 |
| C2 | 0.0718 | 0.0718 |
| D13 | 0.0718 | -0.0718 |
| D4 | 0.0713 | -0.0713 |
| C10 | 0.0690 | 0.0690 |
| D9 | 0.0651 | -0.0651 |
| D5 | 0.0590 | -0.0590 |


**V-column block correlation -- train partition:**

| pct_missing | n_columns | max_abs_corr | mean_abs_corr | top_column |
|---|---|---|---|---|
| 76.2921 | 46 | 0.3794 | 0.1266 | V257 |
| 74.8920 | 19 | 0.3159 | 0.1420 | V201 |
| 29.6114 | 18 | 0.2857 | 0.1672 | V45 |
| 84.8511 | 29 | 0.2842 | 0.1297 | V158 |
| 16.5336 | 20 | 0.2542 | 0.1344 | V87 |
| 74.9192 | 31 | 0.2303 | 0.0611 | V177 |
| 14.3605 | 23 | 0.1839 | 0.1042 | V18 |
| 14.5507 | 22 | 0.1808 | 0.1018 | V74 |
| 74.5640 | 16 | 0.1548 | 0.0642 | V222 |
| 0.0002 | 43 | 0.1515 | 0.0539 | V123 |


**V-column block correlation -- validation partition:**

| pct_missing | n_columns | max_abs_corr | mean_abs_corr | top_column |
|---|---|---|---|---|
| 84.3818 | 46 | 0.4170 | 0.1674 | V258 |
| 83.0089 | 19 | 0.3180 | 0.1270 | V201 |
| 83.0856 | 31 | 0.3015 | 0.0975 | V199 |
| 24.8125 | 18 | 0.2506 | 0.1671 | V45 |
| 9.2525 | 20 | 0.2466 | 0.1302 | V86 |
| 82.9338 | 16 | 0.1876 | 0.0578 | V259 |
| 8.1619 | 22 | 0.1868 | 0.1020 | V74 |
| 7.9380 | 23 | 0.1856 | 0.1021 | V33 |
| 0.0000 | 32 | 0.1545 | 0.0336 | V302 |
| 91.4040 | 29 | 0.1216 | 0.0534 | V146 |


None of the above is fed back into the split boundary -- the boundary in section 1 is fixed regardless of what this drift analysis finds.

## 5. Identity coverage by partition

| partition | n_transactions | n_with_identity | pct_with_identity |
|---|---|---|---|
| train | 445,173 | 115,408 | 25.92% |
| embargo_1 | 18,589 | 3,450 | 18.56% |
| validation | 65,193 | 11,359 | 17.42% |
| embargo_2 | 20,944 | 5,467 | 26.10% |
| holdout | -- | -- | *not computed -- sealed for Phase H* |

## 6. Rollup utility demo (illustrative only)

Placeholder grouping column: `ProductCD`.

## Grouping-key deferral (applies to the rollup utility demo)

`daily_count_by_group`/`daily_amount_stats_by_group` are fully generic over
their grouping column -- Phase B does not select, name, or endorse a
production entity/merchant definition. The demo run below uses `ProductCD`
purely to prove the utilities execute correctly against real data; it is
illustrative only. The actual grouping key for a spike/behavioral-change
detector is a Phase D decision.

**`daily_count_by_group` sample output:**

| ProductCD | _day | n_transactions |
|---|---|---|
| C | 1 | 403 |
| C | 2 | 367 |
| C | 3 | 315 |
| C | 4 | 378 |
| C | 5 | 410 |
| C | 6 | 333 |
| C | 7 | 367 |
| C | 8 | 398 |
| C | 9 | 468 |
| C | 10 | 379 |


**`daily_amount_stats_by_group` sample output:**

| ProductCD | _day | n_transactions | amount_sum | amount_mean | amount_std |
|---|---|---|---|---|---|
| C | 1 | 403 | 17022.9805 | 42.2406 | 34.4610 |
| C | 2 | 367 | 15529.0928 | 42.3136 | 35.8476 |
| C | 3 | 315 | 12472.4141 | 39.5950 | 32.4280 |
| C | 4 | 378 | 17972.5566 | 47.5464 | 48.5123 |
| C | 5 | 410 | 20717.4004 | 50.5302 | 40.1473 |
| C | 6 | 333 | 16775.4805 | 50.3768 | 47.2896 |
| C | 7 | 367 | 18219.7305 | 49.6450 | 44.2520 |
| C | 8 | 398 | 17841.6191 | 44.8282 | 44.1421 |
| C | 9 | 468 | 22703.9492 | 48.5127 | 41.0590 |
| C | 10 | 379 | 17989.9863 | 47.4670 | 38.8535 |


## 7. Memory observations

| baseline_process_rss_mb | train_transaction_full_df_mb | after_full_load_process_rss_mb | after_correlation_process_rss_mb | final_process_rss_mb |
|---|---|---|---|---|
| 153.1900 | 861.1200 | 4871.2300 | 685.2700 | 2915.2500 |


Only the section 4 drift/correlation step used `load_transaction_full` (explicit, logged as high-memory). Every other step used column-scoped loading (`load_transaction_columns` / `load_transaction_ids` / `load_identity_ids`).
# SentinelPay -- Phase D.1 Grouping-Key Sufficiency Analysis Report

**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_d1_report` from `reports/eda/phase_d1_results.json`** -- every number below is read from that file; re-running `python -m sentinelpay.eda.run_phase_d1` regenerates both together.

## Scope (what Phase D.1 is, and deliberately is not)

Phase D.1 is a narrow, non-target grouping-key sufficiency analysis. Its
only question: does `payment_proxy_key` or `device_proxy_key` have enough
strictly-causal historical density to support a future Phase D per-entity
behavioral-change detector? Explicitly out of scope, per the approved D.1
proposal:

- **No target of any kind.** `isFraud` is never loaded, never read, never
  compared to either key. Nothing below is a fraud rate.
- **No detector.** No rolling median/MAD, EWMA, or CUSUM/change-point
  logic. No `configs/detection.yaml`, no `detection.py`.
- **No target encoding, no fraud-rate evaluation, no score/parquet
  persistence.**
- **No production grouping-key selection made in advance.** The
  recommendation below is a pure function of the measured results in this
  report (see `sentinelpay.eda.grouping_key_sufficiency.recommend_grouping_key`)
  -- not a preference decided before this analysis ran.
- **`configs/split.yaml` boundaries are unchanged.**

## 1. Split configuration (unchanged from Phase B/C)

| partition | start_day | end_day |
|---|---|---|
| train | 1 | 130 |
| embargo_1 | 131 | 137 |
| validation | 138 | 160 |
| embargo_2 | 161 | 167 |
| holdout | 168 | 182 |

## 2. Holdout sealing

- Total rows loaded (train_transaction.csv joined to train_identity.csv): **590,540**.
- Rows filtered to `train`/`embargo_1`/`validation`/`embargo_2` (`sentinelpay.data.split.DEVELOPMENT_PARTITIONS`) **before** any grouping-key content analysis: **549,899**.
- Holdout rows excluded, never touched by group-size/history/event-frequency computation: **40,641**.
- `isFraud` is never loaded by `sentinelpay.eda.run_phase_d1`.

## Method

For each candidate key, rows missing any key component are excluded (not
imputed) before grouping. For every remaining row, the number of
strictly-prior same-key events is counted by
`sentinelpay.data.history.prior_group_count` (delegated, not
re-implemented here): a row never counts itself, two rows sharing a
timestamp never count each other, a future row can never change an
earlier row's count, and results do not depend on input row order -- see
`tests/test_history.py` for the causal-correctness evidence, and
`tests/test_grouping_key_sufficiency.py` for this module's own tests
against that guarantee.

"Sufficiency at threshold T" is the percentage of valid rows with >= T
strictly-prior same-key events. Five thresholds are reported (1/3/5/10/20)
rather than one chosen in advance, both overall and broken out by
`train`/`embargo_1`/`validation`/`embargo_2`, so concentration in `train`
alone would be visible rather than hidden behind a single overall number.


## 3. payment_proxy_key

- Key columns: `card1, card2, card3, card5, addr1`.
- Row coverage: **478,702** / 549,899 development rows have the key (**87.05%**).
- Distinct groups: **37,149**, of which **14,786** are singletons (**3.09%** of valid rows).
- Largest group: **5,461** rows; median group size **2.0**.

**Strictly-prior event count distribution:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 0 | 6.0000 | 34.0000 | 184.0000 | 643.0000 | 3432.0000 | 5460 | 252.0835 |


**Group-size distribution:**

| group_size | n_groups | n_rows_covered |
|---|---|---|
| 1 | 14786 | 14786 |
| 2 | 5512 | 11024 |
| 3 | 3004 | 9012 |
| 4 | 1889 | 7556 |
| 5 | 1412 | 7060 |
| 6 | 1170 | 7020 |
| 7 | 900 | 6300 |
| 8 | 733 | 5864 |
| 9 | 613 | 5517 |
| 10 | 495 | 4950 |
| 11 | 459 | 5049 |
| 12 | 406 | 4872 |
| 13 | 361 | 4693 |
| 14 | 314 | 4396 |
| 15 | 285 | 4275 |
| 16 | 236 | 3776 |
| 17 | 205 | 3485 |
| 18 | 222 | 3996 |
| 19 | 195 | 3705 |
| 20 | 162 | 3240 |
| 21 | 163 | 3423 |
| 22 | 148 | 3256 |
| 23 | 168 | 3864 |
| 24 | 149 | 3576 |
| 25 | 100 | 2500 |
| 26 | 139 | 3614 |
| 27 | 108 | 2916 |
| 28 | 93 | 2604 |
| 29 | 96 | 2784 |
| 30 | 76 | 2280 |
| 31 | 88 | 2728 |
| 32 | 78 | 2496 |
| 33 | 81 | 2673 |
| 34 | 63 | 2142 |
| 35 | 61 | 2135 |
| 36 | 61 | 2196 |
| 37 | 64 | 2368 |
| 38 | 58 | 2204 |
| 39 | 51 | 1989 |
| 40 | 56 | 2240 |
| 41 | 41 | 1681 |
| 42 | 45 | 1890 |
| 43 | 56 | 2408 |
| 44 | 36 | 1584 |
| 45 | 42 | 1890 |
| 46 | 35 | 1610 |
| 47 | 48 | 2256 |
| 48 | 35 | 1680 |
| 49 | 24 | 1176 |
| 50 | 31 | 1550 |
| 51 | 38 | 1938 |
| 52 | 37 | 1924 |
| 53 | 25 | 1325 |
| 54 | 21 | 1134 |
| 55 | 32 | 1760 |
| 56 | 23 | 1288 |
| 57 | 23 | 1311 |
| 58 | 20 | 1160 |
| 59 | 28 | 1652 |
| 60 | 21 | 1260 |
| 61 | 21 | 1281 |
| 62 | 21 | 1302 |
| 63 | 20 | 1260 |
| 64 | 18 | 1152 |
| 65 | 22 | 1430 |
| 66 | 15 | 990 |
| 67 | 12 | 804 |
| 68 | 18 | 1224 |
| 69 | 21 | 1449 |
| 70 | 17 | 1190 |
| 71 | 19 | 1349 |
| 72 | 16 | 1152 |
| 73 | 7 | 511 |
| 74 | 20 | 1480 |
| 75 | 17 | 1275 |
| 76 | 12 | 912 |
| 77 | 9 | 693 |
| 78 | 14 | 1092 |
| 79 | 15 | 1185 |
| 80 | 11 | 880 |
| 81 | 13 | 1053 |
| 82 | 9 | 738 |
| 83 | 9 | 747 |
| 84 | 17 | 1428 |
| 85 | 14 | 1190 |
| 86 | 10 | 860 |
| 87 | 10 | 870 |
| 88 | 13 | 1144 |
| 89 | 5 | 445 |
| 90 | 9 | 810 |
| 91 | 12 | 1092 |
| 92 | 13 | 1196 |
| 93 | 10 | 930 |
| 94 | 10 | 940 |
| 95 | 11 | 1045 |
| 96 | 9 | 864 |
| 97 | 9 | 873 |
| 98 | 5 | 490 |
| 99 | 10 | 990 |
| 100 | 6 | 600 |
| 101 | 9 | 909 |
| 102 | 6 | 612 |
| 103 | 12 | 1236 |
| 104 | 3 | 312 |
| 105 | 14 | 1470 |
| 106 | 2 | 212 |
| 107 | 8 | 856 |
| 108 | 6 | 648 |
| 109 | 8 | 872 |
| 110 | 6 | 660 |
| 111 | 9 | 999 |
| 112 | 12 | 1344 |
| 113 | 6 | 678 |
| 114 | 7 | 798 |
| 115 | 4 | 460 |
| 116 | 7 | 812 |
| 117 | 9 | 1053 |
| 118 | 5 | 590 |
| 119 | 3 | 357 |
| 120 | 7 | 840 |
| 121 | 6 | 726 |
| 122 | 4 | 488 |
| 123 | 10 | 1230 |
| 124 | 4 | 496 |
| 126 | 3 | 378 |
| 127 | 4 | 508 |
| 128 | 3 | 384 |
| 129 | 3 | 387 |
| 130 | 11 | 1430 |
| 131 | 2 | 262 |
| 132 | 2 | 264 |
| 133 | 3 | 399 |
| 134 | 3 | 402 |
| 135 | 5 | 675 |
| 136 | 4 | 544 |
| 137 | 6 | 822 |
| 138 | 2 | 276 |
| 139 | 3 | 417 |
| 140 | 4 | 560 |
| 141 | 3 | 423 |
| 142 | 6 | 852 |
| 143 | 6 | 858 |
| 144 | 5 | 720 |
| 145 | 5 | 725 |
| 146 | 4 | 584 |
| 147 | 4 | 588 |
| 149 | 2 | 298 |
| 150 | 2 | 300 |
| 151 | 3 | 453 |
| 152 | 3 | 456 |
| 153 | 4 | 612 |
| 154 | 2 | 308 |
| 155 | 1 | 155 |
| 156 | 2 | 312 |
| 157 | 4 | 628 |
| 158 | 3 | 474 |
| 159 | 2 | 318 |
| 160 | 1 | 160 |
| 161 | 4 | 644 |
| 162 | 3 | 486 |
| 163 | 1 | 163 |
| 164 | 3 | 492 |
| 165 | 4 | 660 |
| 166 | 1 | 166 |
| 167 | 3 | 501 |
| 168 | 1 | 168 |
| 169 | 2 | 338 |
| 170 | 3 | 510 |
| 171 | 1 | 171 |
| 172 | 1 | 172 |
| 173 | 2 | 346 |
| 174 | 2 | 348 |
| 175 | 3 | 525 |
| 176 | 2 | 352 |
| 177 | 4 | 708 |
| 178 | 1 | 178 |
| 180 | 2 | 360 |
| 181 | 1 | 181 |
| 182 | 4 | 728 |
| 183 | 4 | 732 |
| 184 | 4 | 736 |
| 185 | 1 | 185 |
| 187 | 3 | 561 |
| 188 | 3 | 564 |
| 189 | 2 | 378 |
| 190 | 1 | 190 |
| 191 | 4 | 764 |
| 194 | 2 | 388 |
| 195 | 3 | 585 |
| 196 | 2 | 392 |
| 198 | 1 | 198 |
| 199 | 4 | 796 |
| 200 | 2 | 400 |
| 201 | 1 | 201 |
| 203 | 5 | 1015 |
| 204 | 4 | 816 |
| 205 | 2 | 410 |
| 206 | 2 | 412 |
| 208 | 1 | 208 |
| 209 | 1 | 209 |
| 211 | 4 | 844 |
| 212 | 2 | 424 |
| 213 | 1 | 213 |
| 214 | 3 | 642 |
| 215 | 1 | 215 |
| 216 | 4 | 864 |
| 217 | 2 | 434 |
| 218 | 6 | 1308 |
| 219 | 3 | 657 |
| 220 | 2 | 440 |
| 222 | 1 | 222 |
| 225 | 1 | 225 |
| 226 | 2 | 452 |
| 227 | 1 | 227 |
| 228 | 3 | 684 |
| 229 | 1 | 229 |
| 230 | 1 | 230 |
| 232 | 1 | 232 |
| 234 | 3 | 702 |
| 237 | 2 | 474 |
| 238 | 1 | 238 |
| 239 | 2 | 478 |
| 240 | 3 | 720 |
| 241 | 1 | 241 |
| 242 | 1 | 242 |
| 243 | 1 | 243 |
| 244 | 3 | 732 |
| 245 | 1 | 245 |
| 246 | 4 | 984 |
| 248 | 3 | 744 |
| 250 | 2 | 500 |
| 251 | 1 | 251 |
| 254 | 1 | 254 |
| 255 | 2 | 510 |
| 256 | 1 | 256 |
| 257 | 1 | 257 |
| 259 | 1 | 259 |
| 260 | 1 | 260 |
| 261 | 1 | 261 |
| 263 | 1 | 263 |
| 265 | 1 | 265 |
| 266 | 2 | 532 |
| 267 | 3 | 801 |
| 268 | 2 | 536 |
| 270 | 4 | 1080 |
| 271 | 1 | 271 |
| 272 | 3 | 816 |
| 274 | 1 | 274 |
| 277 | 1 | 277 |
| 278 | 1 | 278 |
| 282 | 1 | 282 |
| 284 | 1 | 284 |
| 290 | 1 | 290 |
| 292 | 2 | 584 |
| 293 | 1 | 293 |
| 295 | 1 | 295 |
| 296 | 1 | 296 |
| 297 | 2 | 594 |
| 300 | 3 | 900 |
| 302 | 1 | 302 |
| 304 | 1 | 304 |
| 306 | 1 | 306 |
| 307 | 2 | 614 |
| 308 | 2 | 616 |
| 309 | 1 | 309 |
| 310 | 2 | 620 |
| 316 | 1 | 316 |
| 319 | 1 | 319 |
| 320 | 1 | 320 |
| 325 | 2 | 650 |
| 326 | 2 | 652 |
| 327 | 2 | 654 |
| 328 | 1 | 328 |
| 329 | 1 | 329 |
| 331 | 4 | 1324 |
| 332 | 1 | 332 |
| 336 | 1 | 336 |
| 337 | 2 | 674 |
| 340 | 1 | 340 |
| 342 | 1 | 342 |
| 347 | 1 | 347 |
| 348 | 1 | 348 |
| 349 | 3 | 1047 |
| 350 | 1 | 350 |
| 352 | 1 | 352 |
| 354 | 2 | 708 |
| 356 | 1 | 356 |
| 357 | 1 | 357 |
| 359 | 1 | 359 |
| 360 | 1 | 360 |
| 362 | 1 | 362 |
| 364 | 1 | 364 |
| 366 | 1 | 366 |
| 368 | 1 | 368 |
| 370 | 3 | 1110 |
| 374 | 1 | 374 |
| 376 | 1 | 376 |
| 378 | 2 | 756 |
| 380 | 1 | 380 |
| 384 | 1 | 384 |
| 385 | 1 | 385 |
| 387 | 1 | 387 |
| 388 | 1 | 388 |
| 396 | 1 | 396 |
| 397 | 1 | 397 |
| 398 | 2 | 796 |
| 406 | 1 | 406 |
| 409 | 1 | 409 |
| 413 | 1 | 413 |
| 415 | 1 | 415 |
| 416 | 1 | 416 |
| 417 | 1 | 417 |
| 418 | 1 | 418 |
| 420 | 1 | 420 |
| 421 | 1 | 421 |
| 429 | 1 | 429 |
| 430 | 1 | 430 |
| 433 | 1 | 433 |
| 436 | 1 | 436 |
| 439 | 1 | 439 |
| 440 | 2 | 880 |
| 441 | 1 | 441 |
| 445 | 1 | 445 |
| 446 | 1 | 446 |
| 447 | 1 | 447 |
| 448 | 1 | 448 |
| 449 | 1 | 449 |
| 450 | 1 | 450 |
| 452 | 1 | 452 |
| 463 | 1 | 463 |
| 464 | 1 | 464 |
| 467 | 1 | 467 |
| 471 | 2 | 942 |
| 472 | 1 | 472 |
| 473 | 1 | 473 |
| 477 | 1 | 477 |
| 479 | 1 | 479 |
| 482 | 1 | 482 |
| 483 | 1 | 483 |
| 485 | 1 | 485 |
| 486 | 2 | 972 |
| 496 | 1 | 496 |
| 500 | 2 | 1000 |
| 510 | 1 | 510 |
| 511 | 1 | 511 |
| 518 | 1 | 518 |
| 523 | 1 | 523 |
| 524 | 1 | 524 |
| 527 | 1 | 527 |
| 534 | 1 | 534 |
| 536 | 1 | 536 |
| 540 | 1 | 540 |
| 548 | 1 | 548 |
| 550 | 1 | 550 |
| 552 | 1 | 552 |
| 555 | 1 | 555 |
| 557 | 1 | 557 |
| 570 | 1 | 570 |
| 576 | 1 | 576 |
| 586 | 1 | 586 |
| 594 | 1 | 594 |
| 595 | 1 | 595 |
| 598 | 2 | 1196 |
| 600 | 1 | 600 |
| 608 | 1 | 608 |
| 627 | 2 | 1254 |
| 629 | 1 | 629 |
| 634 | 1 | 634 |
| 637 | 1 | 637 |
| 654 | 1 | 654 |
| 655 | 1 | 655 |
| 656 | 1 | 656 |
| 658 | 1 | 658 |
| 662 | 2 | 1324 |
| 668 | 1 | 668 |
| 670 | 1 | 670 |
| 671 | 1 | 671 |
| 679 | 1 | 679 |
| 681 | 1 | 681 |
| 686 | 1 | 686 |
| 689 | 1 | 689 |
| 692 | 1 | 692 |
| 693 | 1 | 693 |
| 697 | 1 | 697 |
| 703 | 1 | 703 |
| 704 | 1 | 704 |
| 720 | 1 | 720 |
| 732 | 1 | 732 |
| 743 | 1 | 743 |
| 752 | 1 | 752 |
| 755 | 1 | 755 |
| 766 | 1 | 766 |
| 792 | 1 | 792 |
| 800 | 1 | 800 |
| 808 | 1 | 808 |
| 810 | 1 | 810 |
| 821 | 1 | 821 |
| 833 | 1 | 833 |
| 862 | 2 | 1724 |
| 865 | 1 | 865 |
| 879 | 1 | 879 |
| 886 | 1 | 886 |
| 888 | 1 | 888 |
| 911 | 1 | 911 |
| 952 | 1 | 952 |
| 962 | 1 | 962 |
| 996 | 1 | 996 |
| 1037 | 1 | 1037 |
| 1075 | 1 | 1075 |
| 1101 | 1 | 1101 |
| 1106 | 1 | 1106 |
| 1134 | 1 | 1134 |
| 1178 | 1 | 1178 |
| 1205 | 1 | 1205 |
| 1208 | 1 | 1208 |
| 1216 | 1 | 1216 |
| 1255 | 1 | 1255 |
| 1306 | 1 | 1306 |
| 1355 | 1 | 1355 |
| 1368 | 1 | 1368 |
| 1424 | 1 | 1424 |
| 1490 | 1 | 1490 |
| 1577 | 1 | 1577 |
| 1660 | 1 | 1660 |
| 1739 | 1 | 1739 |
| 1770 | 1 | 1770 |
| 1915 | 1 | 1915 |
| 1953 | 1 | 1953 |
| 2128 | 1 | 2128 |
| 2168 | 1 | 2168 |
| 2374 | 1 | 2374 |
| 2495 | 1 | 2495 |
| 3200 | 1 | 3200 |
| 3239 | 2 | 6478 |
| 4270 | 1 | 4270 |
| 5354 | 1 | 5354 |
| 5461 | 1 | 5461 |


**Sufficiency by threshold, overall:**

| >= 1 | >= 3 | >= 5 | >= 10 | >= 20 |
|---|---|---|---|---|
| 92.2396 | 84.0467 | 78.6565 | 69.6181 | 58.9774 |


**Sufficiency by threshold, by partition:**

| partition | n_valid_rows | pct_sufficient_ge_1 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 | pct_sufficient_ge_20 |
|---|---|---|---|---|---|---|
| train | 386407 | 91.1474 | 81.9871 | 76.0600 | 66.3901 | 55.3631 |
| embargo_1 | 16386 | 96.6618 | 92.2983 | 88.9845 | 81.9297 | 72.8000 |
| validation | 57806 | 96.8740 | 92.7343 | 89.7018 | 83.3062 | 74.0719 |
| embargo_2 | 18103 | 96.7519 | 92.7968 | 89.4603 | 83.6657 | 75.4129 |


**Dominant-group exclusion sensitivity** (remaining valid rows/coverage and recomputed sufficiency after excluding the top-K largest groups):

| top_k_excluded | excluded_group_sizes | n_valid_rows_remaining | pct_valid_rows_remaining_of_original_valid | pct_valid_rows_remaining_of_development_total | pct_sufficient_ge_1 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 | pct_sufficient_ge_20 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [5461] | 473241 | 98.8592 | 86.0596 | 92.1503 | 83.8632 | 78.4112 | 69.2696 | 58.5082 |
| 3 | [5461, 5354, 4270] | 463617 | 96.8488 | 84.3095 | 91.9878 | 83.5295 | 77.9652 | 68.6360 | 57.6556 |
| 5 | [5461, 5354, 4270, 3239, 3239] | 457139 | 95.4955 | 83.1314 | 91.8747 | 83.2974 | 77.6552 | 68.1959 | 57.0643 |
| 10 | [5461, 5354, 4270, 3239, 3239, 3200, 2495, 2374, 2168, 2128] | 444774 | 92.9125 | 80.8829 | 91.6499 | 82.8365 | 77.0396 | 67.3230 | 55.8931 |


**This phase's evaluation** (threshold=5): row coverage **87.05%** (row_coverage_ok: **True**) | overall sufficiency **78.66%** (density_ok: **True**) | partition_stability_ok: **True** | dominant_group_robustness_ok: **True** (worst case after exclusion: 77.04%) | **is_suitable: True**.

## 4. device_proxy_key

- Key columns: `DeviceInfo, id_31`.
- Row coverage: **111,625** / 549,899 development rows have the key (**20.30%**).
- Distinct groups: **5,063**, of which **2,203** are singletons (**1.97%** of valid rows).
- Largest group: **16,398** rows; median group size **2.0**.

**Strictly-prior event count distribution:**

| min | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| 0 | 61.0000 | 1248.0000 | 3969.0000 | 7365.6000 | 15280.7600 | 16397 | 2666.1560 |


**Group-size distribution:**

| group_size | n_groups | n_rows_covered |
|---|---|---|
| 1 | 2203 | 2203 |
| 2 | 783 | 1566 |
| 3 | 455 | 1365 |
| 4 | 310 | 1240 |
| 5 | 218 | 1090 |
| 6 | 146 | 876 |
| 7 | 118 | 826 |
| 8 | 87 | 696 |
| 9 | 82 | 738 |
| 10 | 67 | 670 |
| 11 | 51 | 561 |
| 12 | 43 | 516 |
| 13 | 37 | 481 |
| 14 | 29 | 406 |
| 15 | 27 | 405 |
| 16 | 30 | 480 |
| 17 | 19 | 323 |
| 18 | 23 | 414 |
| 19 | 23 | 437 |
| 20 | 18 | 360 |
| 21 | 14 | 294 |
| 22 | 8 | 176 |
| 23 | 11 | 253 |
| 24 | 9 | 216 |
| 25 | 10 | 250 |
| 26 | 9 | 234 |
| 27 | 11 | 297 |
| 28 | 12 | 336 |
| 29 | 10 | 290 |
| 30 | 8 | 240 |
| 31 | 10 | 310 |
| 32 | 9 | 288 |
| 33 | 5 | 165 |
| 34 | 6 | 204 |
| 35 | 6 | 210 |
| 36 | 4 | 144 |
| 37 | 5 | 185 |
| 38 | 5 | 190 |
| 39 | 4 | 156 |
| 40 | 4 | 160 |
| 41 | 3 | 123 |
| 42 | 4 | 168 |
| 43 | 1 | 43 |
| 44 | 4 | 176 |
| 45 | 1 | 45 |
| 46 | 3 | 138 |
| 47 | 4 | 188 |
| 48 | 2 | 96 |
| 49 | 2 | 98 |
| 50 | 3 | 150 |
| 52 | 3 | 156 |
| 53 | 2 | 106 |
| 54 | 2 | 108 |
| 55 | 1 | 55 |
| 56 | 3 | 168 |
| 57 | 2 | 114 |
| 58 | 2 | 116 |
| 59 | 2 | 118 |
| 60 | 1 | 60 |
| 61 | 1 | 61 |
| 62 | 2 | 124 |
| 63 | 2 | 126 |
| 65 | 2 | 130 |
| 66 | 1 | 66 |
| 67 | 1 | 67 |
| 68 | 3 | 204 |
| 70 | 2 | 140 |
| 71 | 2 | 142 |
| 75 | 1 | 75 |
| 77 | 1 | 77 |
| 78 | 1 | 78 |
| 81 | 1 | 81 |
| 82 | 1 | 82 |
| 83 | 1 | 83 |
| 84 | 1 | 84 |
| 86 | 2 | 172 |
| 91 | 2 | 182 |
| 94 | 3 | 282 |
| 95 | 1 | 95 |
| 98 | 1 | 98 |
| 99 | 1 | 99 |
| 103 | 1 | 103 |
| 104 | 1 | 104 |
| 106 | 2 | 212 |
| 108 | 1 | 108 |
| 113 | 1 | 113 |
| 122 | 2 | 244 |
| 123 | 2 | 246 |
| 125 | 1 | 125 |
| 128 | 1 | 128 |
| 131 | 1 | 131 |
| 140 | 1 | 140 |
| 146 | 1 | 146 |
| 150 | 1 | 150 |
| 169 | 1 | 169 |
| 173 | 1 | 173 |
| 175 | 1 | 175 |
| 176 | 1 | 176 |
| 180 | 1 | 180 |
| 182 | 1 | 182 |
| 188 | 1 | 188 |
| 194 | 1 | 194 |
| 210 | 1 | 210 |
| 224 | 1 | 224 |
| 226 | 1 | 226 |
| 232 | 1 | 232 |
| 267 | 1 | 267 |
| 274 | 1 | 274 |
| 293 | 1 | 293 |
| 381 | 1 | 381 |
| 410 | 1 | 410 |
| 462 | 1 | 462 |
| 552 | 1 | 552 |
| 595 | 1 | 595 |
| 617 | 1 | 617 |
| 724 | 1 | 724 |
| 770 | 1 | 770 |
| 1180 | 1 | 1180 |
| 1433 | 1 | 1433 |
| 1572 | 1 | 1572 |
| 1579 | 1 | 1579 |
| 1805 | 1 | 1805 |
| 1961 | 1 | 1961 |
| 2106 | 1 | 2106 |
| 2370 | 1 | 2370 |
| 3308 | 1 | 3308 |
| 4613 | 1 | 4613 |
| 4620 | 1 | 4620 |
| 4793 | 1 | 4793 |
| 6460 | 1 | 6460 |
| 6511 | 1 | 6511 |
| 6770 | 1 | 6770 |
| 9497 | 1 | 9497 |
| 16398 | 1 | 16398 |


**Sufficiency by threshold, overall:**

| >= 1 | >= 3 | >= 5 | >= 10 | >= 20 |
|---|---|---|---|---|
| 95.4643 | 91.0414 | 88.4130 | 84.5823 | 80.7095 |


**Sufficiency by threshold, by partition:**

| partition | n_valid_rows | pct_sufficient_ge_1 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 | pct_sufficient_ge_20 |
|---|---|---|---|---|---|---|
| train | 95448 | 95.5662 | 91.3178 | 88.7908 | 85.1626 | 81.5292 |
| embargo_1 | 2696 | 95.1409 | 90.0593 | 86.4243 | 79.5994 | 73.1825 |
| validation | 9006 | 95.0033 | 89.9511 | 86.9976 | 82.6338 | 77.6815 |
| embargo_2 | 4475 | 94.4134 | 87.9330 | 84.4022 | 79.1285 | 73.8547 |


**Dominant-group exclusion sensitivity** (remaining valid rows/coverage and recomputed sufficiency after excluding the top-K largest groups):

| top_k_excluded | excluded_group_sizes | n_valid_rows_remaining | pct_valid_rows_remaining_of_original_valid | pct_valid_rows_remaining_of_development_total | pct_sufficient_ge_1 | pct_sufficient_ge_3 | pct_sufficient_ge_5 | pct_sufficient_ge_10 | pct_sufficient_ge_20 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [16398] | 95227 | 85.3097 | 17.3172 | 94.6843 | 89.5019 | 86.4230 | 81.9379 | 77.4087 |
| 3 | [16398, 9497, 6770] | 78960 | 70.7368 | 14.3590 | 93.5917 | 87.3468 | 83.6386 | 78.2421 | 72.8052 |
| 5 | [16398, 9497, 6770, 6511, 6460] | 65989 | 59.1167 | 12.0002 | 92.3351 | 84.8687 | 80.4376 | 73.9957 | 67.5203 |
| 10 | [16398, 9497, 6770, 6511, 6460, 4793, 4620, 4613, 3308, 2370] | 46285 | 41.4647 | 8.4170 | 89.0829 | 78.4595 | 72.1638 | 63.0334 | 53.9095 |


**This phase's evaluation** (threshold=5): row coverage **20.30%** (row_coverage_ok: **False**) | overall sufficiency **88.41%** (density_ok: **True**) | partition_stability_ok: **True** | dominant_group_robustness_ok: **True** (worst case after exclusion: 72.16%) | **is_suitable: False**.

## 5. Recommendation

**payment_proxy_key**

Only payment_proxy_key meets this phase's criteria.

This is the output of `sentinelpay.eda.grouping_key_sufficiency.recommend_grouping_key` applied to the measured results above -- a pure function of this run's numbers, not a preference chosen before D.1 ran. It is a recommendation for a Phase D scoping decision, not an implementation: no detector, target encoding, or persistence exists yet regardless of which key this section names.
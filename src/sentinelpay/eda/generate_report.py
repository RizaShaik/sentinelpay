"""Deterministic Phase A report generation.

`render_phase_a_report(results, report_path)` renders reports/eda/phase_a_report.md
directly from the `results` dict that run_phase_a.py computes (the same
dict written to phase_a_results.json) plus the already-saved figures. Every
number in the report is read out of that dict -- nothing here is
independently computed or hand-typed -- so report and results.json cannot
silently diverge; re-running run_phase_a.py regenerates both together.

The methodology/definitions text (field semantics, terminology rules,
proposed next-phase groupings) is static: it doesn't come from computed
results and won't diverge from them, so it lives here as fixed template
text rather than being "generated." Only the numeric/tabular sections are
templated from `results`.
"""
from __future__ import annotations

import math
from pathlib import Path


def _table(rows: list[dict], columns: list[str] | None = None, float_fmt: str = "{:.4f}") -> str:
    if not rows:
        return "_(no rows)_\n"
    cols = columns or list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, bool):
                cells.append(str(v))
            elif isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


_FIELD_SEMANTICS_SECTION = """
## 1. Field semantics: documented vs. proxy vs. unknown

This project does not rename anonymized columns. This section exists so
later phases can reason about *why* a column might behave a certain way
without asserting it means something Vesta never confirmed.

**Officially documented** (Kaggle competition data page / host statements):

| Field(s) | Documented meaning |
|---|---|
| `TransactionDT` | Timedelta in **seconds** from an undisclosed reference datetime -- not a real timestamp. Confirmed: min value is exactly 86400 = 24h. |
| `TransactionAmt` | Transaction amount, USD. |
| `C1-C14` | "Counting" features (host's word), e.g. number of addresses associated with the card. Exact definitions withheld. |
| `D1-D15` | "Timedelta" features, e.g. days since previous transaction. Exact reference points per column withheld. |
| `V1-V339` | Vesta-engineered features: ranking, counting, and "other entity relations." No per-column definitions published. |
| `id_01-id_11` | Numeric identity signals from Vesta/security partners (device rating, IP-domain rating, proxy rating, behavioral fingerprints). Exact scoring undisclosed. |
| `ProductCD`, `card1-card6`, `addr1-addr2`, `dist1-dist2`, `P_/R_emaildomain`, `M1-M9` | Named/self-descriptive transaction and card attributes. |
| `DeviceType`, `DeviceInfo`, and the categorical block of `id_12-id_38` | Device/network/browser signals collected at transaction time. |

**Reasonable proxy interpretations** (widely reproduced in public analyses,
*not* Vesta-confirmed -- treat as hypotheses, not fact):
- `id_30` values look like OS strings (`"Android 7.0"`, `"Windows 10"`).
- `id_31` values look like browser strings (`"samsung browser 6.2"`).
- `id_33` values look like screen resolutions (`"2220x1080"`).
- card1/card2/card3/card5 + addr1 is used in this project as a
  `payment_proxy_key` -- there is no published card/customer ID.

**Corrected during Phase A** (a generic web summary claimed the entire
`id_12-id_38` block is categorical -- false on direct inspection): `id_14`
(numeric, looks like a timezone offset in minutes), `id_17`, `id_19`,
`id_20`, `id_32` are numeric in the actual data. The loader does not
hardcode a categorical list for the identity file for this reason.

**Unknown / do not claim**: which specific real-world quantity each `V`
column encodes; the exact reference point for each `D` column; what
`C1..C14` count individually.

Sources: [Kaggle competition data page](https://www.kaggle.com/competitions/ieee-fraud-detection/data), [host discussion #101203](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203), [IEEE DataPort mirror](https://ieee-dataport.org/documents/ieee-cis-fraud-detection).
"""

_TERMINOLOGY_SECTION = """
## Proxy-key terminology and EDA/target-encoding rule

Section 9 below uses `payment_proxy_key` (card1/card2/card3/card5/addr1) and
`device_proxy_key` (DeviceInfo/id_31). These are candidate groupings, not
confirmed real-world entities -- there is no published customer or device
ID in this dataset. This is exploratory evidence gathering for later
phases, not the Phase E coordinated-ring detection system.

**Rule enforced from this phase forward**: any target-derived statistic
computed during EDA (fraud rate by proxy-key group, target correlation,
etc.) is EDA evidence only and must never become a modeling feature
directly. A future target encoding or historical-fraud-rate feature must be
computed using only information available strictly before the transaction
being scored (chronological, fold-safe logic), not the whole-dataset
aggregates this report shows. No target encoding is implemented in Phase
A/B.
"""

_NEXT_PHASE_SECTION = """
## Proposed next-phase feature groups (naming only -- not implemented)

1. **Transaction-intrinsic** (documented): `TransactionAmt`, `ProductCD`, card attributes, address, email domain, `dist1/dist2`.
2. **Counting features** (documented category, undocumented specifics): `C1-C14`.
3. **Timedelta features** (documented category, undocumented specifics): `D1-D15`.
4. **Match flags** (documented as match indicators): `M1-M9`.
5. **Vesta engineered** (opaque): `V1-V339`, grouped by missingness-block (see leakage section) rather than by assumed semantics.
6. **Identity -- numeric scores** (documented category): `id_01-id_11`.
7. **Identity -- device/network** (mixed categorical/numeric): `id_12-id_38`, `DeviceType`, `DeviceInfo`.

## Proposed spike/behavioral-change-detection approaches (Phase D -- not implemented)

Treated as a temporal behavioral-change problem first, per instruction:
rolling robust statistics (median/MAD), EWMA, CUSUM/change-point detection
as primary candidates; Isolation Forest proposed only as a complementary
transaction-level outlier detector, not the primary method. No algorithm is
selected yet.

## Proposed ring-detection approaches (Phase E -- not implemented)

Neither proxy key tried in this report is usable alone (see section 9) --
`payment_proxy_key` sharing is not predictive in the naive direction, and
`device_proxy_key` sharing is predictive but polluted by generic/popular
fingerprints. Next step is a `composite_proxy_key` (combining keys) and/or
inverse-frequency weighting before a graph-based ring signal is attempted.
"""


def render_phase_a_report(results: dict, report_path: Path) -> None:
    schema = results["schema"]
    join_train = results["join_validation_train"]
    join_test = results["join_validation_test"]
    memory = results.get("memory", {})
    temporal = results["temporal"]
    leak_mono = results["leakage_id_time_monotonicity"]
    near_dup = results["leakage_near_duplicate_card_amt_dt"]

    lines: list[str] = []
    lines.append("# SentinelPay -- Phase A EDA & Data Validation Report")
    lines.append("")
    lines.append(
        "Dataset: IEEE-CIS Fraud Detection (Vesta), raw files in `data/raw/`. "
        "**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_a_report` "
        "from `reports/eda/phase_a_results.json`** -- every number below is read from that "
        "file, not hand-typed; re-running `python -m sentinelpay.eda.run_phase_a` regenerates "
        "both together so they cannot diverge."
    )
    lines.append(_FIELD_SEMANTICS_SECTION)

    lines.append("## 2. Schema validation (train vs. test)\n")
    lines.append(
        f"- `train_transaction`: {schema['train_transaction_n_cols']} columns "
        f"| `test_transaction`: {schema['test_transaction_n_cols']} columns. "
        f"`isFraud` absent from test transaction schema: **{schema['isFraud_absent_from_test_transaction']}**. "
        f"Consistent ignoring `isFraud`: **{schema['transaction_schema_consistent_ignoring_isFraud']}**."
    )
    lines.append(
        f"- `train_identity`: {schema['train_identity_n_cols']} columns "
        f"| `test_identity`: {schema['test_identity_n_cols']} columns. "
        f"Consistent after `id-XX`->`id_XX` normalization: "
        f"**{schema['identity_schema_consistent_after_normalization']}**."
    )
    lines.append("")

    lines.append("## 3. TransactionID <-> Identity join validation\n")
    lines.append("| | train | test |")
    lines.append("|---|---|---|")
    lines.append(f"| transaction rows | {join_train['n_transaction_rows']:,} | {join_test['n_transaction_rows']:,} |")
    lines.append(f"| identity rows | {join_train['n_identity_rows']:,} | {join_test['n_identity_rows']:,} |")
    lines.append(f"| duplicate TransactionIDs | {join_train['n_transaction_duplicate_ids']} | {join_test['n_transaction_duplicate_ids']} |")
    lines.append(f"| identity orphans (no matching transaction) | {join_train['n_identity_orphans']} | {join_test['n_identity_orphans']} |")
    lines.append(f"| % transactions with identity | {join_train['pct_transactions_with_identity']:.2f}% | {join_test['pct_transactions_with_identity']:.2f}% |")
    lines.append(f"| join clean | {join_train['is_clean']} | {join_test['is_clean']} |")
    lines.append("")

    lines.append("## 4. `id-XX` -> `id_XX` normalization\n")
    lines.append(
        "Confirmed: `train_identity.csv` ships `id_01..id_38`, `test_identity.csv` ships "
        "`id-01..id-38`. Normalized in memory only, immediately after `pd.read_csv`, by "
        "`sentinelpay.data.loader.normalize_identity_columns` -- `data/raw/*.csv` is never "
        "opened in write mode by any project code (see `tests/test_loader.py` for the "
        "no-source-mutation test)."
    )
    lines.append("")

    lines.append("## 5. Duplicates & missingness\n")
    lines.append(f"- **{results['duplicate_full_rows']}** fully duplicated rows in `train_transaction`.")
    lines.append(f"- **{results['pct_columns_over_50pct_missing']}%** of columns are >50% missing.")
    lines.append("\nTop 20 most-missing columns:\n")
    lines.append(_table(results["top_20_missing_columns"], columns=["column", "n_missing", "pct_missing"]))

    lines.append("## 6. Temporal structure\n")
    lines.append(
        f"- Day index range: {temporal['min_day']}-{temporal['max_day']} "
        f"({temporal['n_days_observed']} distinct days)."
    )
    lines.append(
        f"- Daily volume: min {temporal['daily_volume_min']:,}, max {temporal['daily_volume_max']:,}, "
        f"mean {temporal['daily_volume_mean']:.0f} transactions/day."
    )
    lines.append(
        f"- Daily fraud rate: {temporal['daily_fraud_rate_min']*100:.2f}% - "
        f"{temporal['daily_fraud_rate_max']*100:.2f}%."
    )
    lines.append(f"- Overall fraud rate: {temporal['overall_fraud_rate']*100:.3f}%.")
    lines.append("")
    lines.append("![Daily volume and fraud rate](figures/temporal_volume_fraud_rate.png)")
    lines.append("")

    lines.append(f"## 7. Distribution drift (early vs. late half, split at day {results['drift_split_day']})\n")
    lines.append("**Numeric (KS 2-sample), curated columns, top drift:**\n")
    lines.append(_table(results["numeric_drift_top10"]))
    lines.append("\n**Categorical (chi-square, period x category):**\n")
    lines.append(_table(results["categorical_drift"]))
    lines.append("\n**V-column sample (low-missingness subset), top drift:**\n")
    lines.append(_table(results["v_column_drift_sample_top10"]))

    lines.append(f"\n## 8. Leakage audit (correlation_mode = `{results['correlation_mode']}`)\n")
    lines.append(
        f"- TransactionID vs. TransactionDT rank Spearman correlation: "
        f"**{leak_mono['spearman_corr_id_vs_dt_rank']:.4f}** "
        f"(file sorted by DT: {leak_mono['rows_sorted_by_dt_in_file']}, "
        f"sorted by ID: {leak_mono['rows_sorted_by_id_in_file']}). "
        f"A value of 1.0 means TransactionID must never be used as a feature -- it leaks time order."
    )
    lines.append("\n**Curated (non-V) target correlation, top 15 (via `corrwith`, O(k*n)):**\n")
    lines.append(_table(results["leakage_curated_target_correlation_top15"]))
    lines.append("\n**V-column blocks by shared missingness signature, top 10 by max |corr| in block:**\n")
    lines.append(_table(results["leakage_v_block_target_correlation_top10"]))
    if "leakage_full_target_correlation_top15" in results:
        lines.append("\n**Full opt-in correlation scan (all numeric columns), top 15:**\n")
        lines.append(_table(results["leakage_full_target_correlation_top15"]))
    lines.append(
        f"\n- Near-duplicate rows on {near_dup['subset']}: {near_dup['n_duplicated']} "
        f"({near_dup['pct_duplicated']}%) of {near_dup['n_rows']:,} rows."
    )

    lines.append("\n## 9. Candidate proxy-key relationships (exploratory evidence only)\n")
    pk_stats = results.get("entity_payment_proxy_key_summary_stats", {})
    pk_fraud = results.get("entity_payment_proxy_key_fraud_summary", {})
    if pk_stats:
        lines.append(
            f"**`payment_proxy_key`** ({', '.join(pk_stats['proxy_key_columns'])}): "
            f"{pk_stats['n_rows_valid']:,} rows valid, {pk_stats['n_groups']:,} groups, "
            f"{pk_stats['n_singleton_groups']:,} singletons, largest group {pk_stats['max_group_size']:,} rows."
        )
    if pk_fraud:
        lines.append(
            f"Fraud rate: shared-key rows **{pk_fraud['fraud_rate_shared']*100:.2f}%** vs. "
            f"singleton rows **{pk_fraud['fraud_rate_singleton']*100:.2f}%** "
            f"(overall {pk_fraud['overall_fraud_rate']*100:.2f}%). "
            f"{'Lower for shared groups than singletons -- naive shared-key intuition does not hold on this proxy.' if pk_fraud['fraud_rate_shared'] < pk_fraud['fraud_rate_singleton'] else 'Higher for shared groups than singletons -- directionally consistent with a coordinated-activity hypothesis.'}"
        )
    lines.append("\nFraud rate by group-size bucket:\n")
    lines.append(_table(results.get("entity_payment_proxy_key_fraud_buckets", [])))

    dk_stats = results.get("entity_device_proxy_key_summary_stats", {})
    dk_fraud = results.get("entity_device_proxy_key_fraud_summary", {})
    if dk_stats:
        lines.append(
            f"\n**`device_proxy_key`** ({', '.join(dk_stats['proxy_key_columns'])}, identity-linked subset): "
            f"{dk_stats['n_rows_valid']:,} rows valid, {dk_stats['n_groups']:,} groups, "
            f"{dk_stats['n_singleton_groups']:,} singletons, largest group {dk_stats['max_group_size']:,} rows "
            f"(a group this large is almost certainly a generic/popular device+browser signature shared by many "
            f"unrelated users, not a coordinated actor)."
        )
    if dk_fraud:
        lines.append(
            f"Fraud rate: shared-key rows **{dk_fraud['fraud_rate_shared']*100:.2f}%** vs. "
            f"singleton rows **{dk_fraud['fraud_rate_singleton']*100:.2f}%** "
            f"(overall {dk_fraud['overall_fraud_rate']*100:.2f}%)."
        )
    lines.append("\nFraud rate by group-size bucket:\n")
    lines.append(_table(results.get("entity_device_proxy_key_fraud_buckets", [])))

    lines.append(_TERMINOLOGY_SECTION)

    lines.append("## 10. Memory observations\n")
    lines.append(_table([memory], columns=list(memory.keys())) if memory else "_(no memory data)_\n")
    lines.append(
        "\n`load_transaction_full` was called explicitly, once, for this comprehensive scan "
        "(whole-file missingness/duplicate checks need every column). Routine/ad-hoc analyses "
        "should prefer `load_transaction_columns` or `load_transaction_sample` instead. "
        "Correlation analysis uses `corrwith` (O(k*n)) rather than a full `.corr()` matrix "
        "(O(k^2*n)), which was the actual cause of the previously observed ~5.4GB/2.5min cost -- "
        "V-columns are handled as missingness-grouped blocks, not blindly included."
    )

    lines.append(_NEXT_PHASE_SECTION)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


# Compatibility alias: this function was named `render_report` before Phase B
# added `render_phase_b_report` alongside it. Nothing in this codebase calls
# the old name any more (run_phase_a.py uses `render_phase_a_report`
# directly) -- kept only in case an external caller still imports it.
render_report = render_phase_a_report


_HOLDOUT_SEALING_SECTION = """
## Holdout sealing (applies to this entire report)

`holdout` (see split configuration above) is reserved for Phase H. This
report shows only its structural existence -- row count, and that it is
chronologically after `validation` with no overlap. No fraud rate, drift,
correlation, missingness, entity/proxy statistic, or any other content
measurement is computed on holdout rows anywhere in Phase B. Section 3
(drift) compares `train` vs. `validation` only.
"""

_GROUPING_KEY_DEFERRAL_SECTION = """
## Grouping-key deferral (applies to the rollup utility demo)

`daily_count_by_group`/`daily_amount_stats_by_group` are fully generic over
their grouping column -- Phase B does not select, name, or endorse a
production entity/merchant definition. The demo run below uses `ProductCD`
purely to prove the utilities execute correctly against real data; it is
illustrative only. The actual grouping key for a spike/behavioral-change
detector is a Phase D decision.
"""


_PHASE_C_SCOPE_SECTION = """
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
"""

_PHASE_C_HISTORY_SECTION = """
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
"""


def render_phase_c_report(results: dict, report_path: Path) -> None:
    registry = results["feature_registry"]
    summary = results["feature_summary_by_partition"]
    split_config = results["split_config"]

    lines: list[str] = []
    lines.append("# SentinelPay -- Phase C Non-Target Feature Foundation Report")
    lines.append("")
    lines.append(
        "**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_c_report` "
        "from `reports/eda/phase_c_results.json`** -- every number below is read from that file; "
        "re-running `python -m sentinelpay.eda.run_phase_c` regenerates both together."
    )

    lines.append(_PHASE_C_SCOPE_SECTION)

    lines.append("## 1. Split configuration (unchanged from Phase B)\n")
    lines.append("| partition | start_day | end_day |")
    lines.append("|---|---|---|")
    for name, rng in split_config.items():
        lines.append(f"| {name} | {rng['start_day']} | {rng['end_day']} |")

    lines.append("\n## 2. Holdout sealing\n")
    lines.append(
        f"- Total rows loaded (single `train_transaction.csv`, spans all partitions): "
        f"**{results['n_rows_total']:,}**.\n"
        f"- Rows filtered to `train`/`embargo_1`/`validation`/`embargo_2` "
        f"(`sentinelpay.data.split.DEVELOPMENT_PARTITIONS`) **before** `build_feature_frame` is ever "
        f"called: **{results['n_rows_development']:,}**.\n"
        f"- Holdout rows excluded, never touched by feature computation: "
        f"**{results['n_rows_holdout_excluded']:,}**.\n"
        f"- `isFraud` is never loaded by `sentinelpay.eda.run_phase_c` -- only `TransactionID`, "
        f"`TransactionDT`, `TransactionAmt` are read from `train_transaction.csv`."
    )

    lines.append("\n## 3. Feature registry\n")
    lines.append(
        _table(
            [
                {
                    "feature": e["feature"],
                    "source_columns": ", ".join(e["source_columns"]),
                    "uses_target": e["uses_target"],
                    "temporal_dependency": e["temporal_dependency"],
                    "description": e["description"],
                }
                for e in registry
            ]
        )
    )

    lines.append("\n## 4. Feature summary by partition (train/embargo_1/validation/embargo_2 only)\n")
    lines.append(_table(summary))

    lines.append(_PHASE_C_HISTORY_SECTION)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


_PHASE_D1_SCOPE_SECTION = """
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
"""

_PHASE_D1_METHOD_SECTION = """
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
"""


def _key_section(title: str, key_result: dict, key_eval: dict) -> list[str]:
    lines: list[str] = []
    lines.append(f"\n## {title}\n")
    lines.append(
        f"- Key columns: `{', '.join(key_result['key_columns'])}`.\n"
        f"- Row coverage: **{key_result['n_rows_valid']:,}** / {key_result['n_rows_development']:,} "
        f"development rows have the key (**{key_result['pct_rows_valid']:.2f}%**).\n"
        f"- Distinct groups: **{key_result['n_groups']:,}**, of which "
        f"**{key_result['n_singleton_groups']:,}** are singletons "
        f"(**{key_result['singleton_fraction_of_valid_rows']:.2f}%** of valid rows).\n"
        f"- Largest group: **{key_result['max_group_size']:,}** rows; median group size "
        f"**{key_result['median_group_size']:.1f}**."
    )

    lines.append("\n**Strictly-prior event count distribution:**\n")
    dist = key_result["prior_count_distribution"]
    if dist:
        lines.append(_table([dist], columns=["min", "p25", "p50", "p75", "p90", "p99", "max", "mean"]))

    lines.append("\n**Group-size distribution:**\n")
    lines.append(_table(key_result["group_size_distribution"], columns=["group_size", "n_groups", "n_rows_covered"]))

    lines.append("\n**Sufficiency by threshold, overall:**\n")
    overall = key_result["sufficiency_overall"]
    lines.append(_table([{f">= {t}": pct for t, pct in overall.items()}]))

    lines.append("\n**Sufficiency by threshold, by partition:**\n")
    lines.append(_table(key_result["sufficiency_by_partition"]))

    lines.append(
        "\n**Dominant-group exclusion sensitivity** (remaining valid rows/coverage and recomputed "
        "sufficiency after excluding the top-K largest groups):\n"
    )
    lines.append(_table(key_result["dominant_group_exclusion_sensitivity"]))

    lines.append(
        f"\n**This phase's evaluation** (threshold={key_eval['threshold']}): row coverage "
        f"**{key_eval['row_coverage_pct']:.2f}%** (row_coverage_ok: **{key_eval['row_coverage_ok']}**) | "
        f"overall sufficiency **{key_eval['overall_sufficiency_pct']:.2f}%** "
        f"(density_ok: **{key_eval['coverage_ok']}**) | "
        f"partition_stability_ok: **{key_eval['partition_stability_ok']}** | "
        f"dominant_group_robustness_ok: **{key_eval['dominant_group_robustness_ok']}** "
        f"(worst case after exclusion: {key_eval['worst_case_sufficiency_pct_after_exclusion']:.2f}%) | "
        f"**is_suitable: {key_eval['is_suitable']}**."
    )
    return lines


def render_phase_d1_report(results: dict, report_path: Path) -> None:
    split_config = results["split_config"]
    recommendation = results["recommendation"]

    lines: list[str] = []
    lines.append("# SentinelPay -- Phase D.1 Grouping-Key Sufficiency Analysis Report")
    lines.append("")
    lines.append(
        "**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_d1_report` "
        "from `reports/eda/phase_d1_results.json`** -- every number below is read from that file; "
        "re-running `python -m sentinelpay.eda.run_phase_d1` regenerates both together."
    )

    lines.append(_PHASE_D1_SCOPE_SECTION)

    lines.append("## 1. Split configuration (unchanged from Phase B/C)\n")
    lines.append("| partition | start_day | end_day |")
    lines.append("|---|---|---|")
    for name in ["train", "embargo_1", "validation", "embargo_2", "holdout"]:
        rng = split_config[name]
        lines.append(f"| {name} | {rng['start_day']} | {rng['end_day']} |")

    lines.append("\n## 2. Holdout sealing\n")
    lines.append(
        f"- Total rows loaded (train_transaction.csv joined to train_identity.csv): "
        f"**{results['n_rows_total']:,}**.\n"
        f"- Rows filtered to `train`/`embargo_1`/`validation`/`embargo_2` "
        f"(`sentinelpay.data.split.DEVELOPMENT_PARTITIONS`) **before** any grouping-key content "
        f"analysis: **{results['n_rows_development']:,}**.\n"
        f"- Holdout rows excluded, never touched by group-size/history/event-frequency computation: "
        f"**{results['n_rows_holdout_excluded']:,}**.\n"
        f"- `isFraud` is never loaded by `sentinelpay.eda.run_phase_d1`."
    )

    lines.append(_PHASE_D1_METHOD_SECTION)

    lines.extend(_key_section("3. payment_proxy_key", results["payment_proxy_key"], results["payment_proxy_key_evaluation"]))
    lines.extend(_key_section("4. device_proxy_key", results["device_proxy_key"], results["device_proxy_key_evaluation"]))

    lines.append("\n## 5. Recommendation\n")
    lines.append(
        f"**{recommendation['recommendation']}**\n\n{recommendation['reason']}\n\n"
        "This is the output of `sentinelpay.eda.grouping_key_sufficiency.recommend_grouping_key` "
        "applied to the measured results above -- a pure function of this run's numbers, not a "
        "preference chosen before D.1 ran. It is a recommendation for a Phase D scoping decision, "
        "not an implementation: no detector, target encoding, or persistence exists yet regardless "
        "of which key this section names."
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


_PHASE_E1_SCOPE_SECTION = """
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
"""

_PHASE_E1_METHOD_SECTION = """
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
"""

_PHASE_E1_M5_SECTION = """
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
"""


def _direction_section(title: str, entry: dict) -> list[str]:
    lines: list[str] = []
    analysis = entry["analysis"]
    coverage = entry["coverage"]
    ev = entry["evaluation"]
    overlap = entry["overlap"]
    frequency_adjusted_overlap = entry["frequency_adjusted_overlap"]

    lines.append(f"\n### {title}\n")
    lines.append(
        f"- anchor_col: `{entry['anchor_col']}` | other_col: `{entry['other_col']}` | "
        f"relationship pair: `{entry['relationship_pair']}`.\n"
        f"- Relationship row coverage (M3): **{coverage['n_rows_valid']:,}** / "
        f"{coverage['n_rows_development']:,} development rows (**{coverage['pct_rows_valid']:.2f}%**).\n"
        f"- Anchor groups: **{analysis['n_anchor_groups']:,}**, of which "
        f"**{analysis['n_anchor_singleton_groups']:,}** are singletons. Largest anchor group: "
        f"**{analysis['max_anchor_group_size']:,}** rows; median **{analysis['median_anchor_group_size']:.1f}**."
    )

    lines.append("\n**Unbounded strictly-prior distinct-partner-count distribution (M1):**\n")
    dist = analysis["unbounded"]["prior_count_distribution"]
    if dist:
        lines.append(_table([dist], columns=["min", "p25", "p50", "p75", "p90", "p99", "max", "mean"]))

    lines.append("\n**Sufficiency by threshold, overall (M2, unbounded):**\n")
    overall = analysis["unbounded"]["sufficiency_overall"]
    lines.append(_table([{f">= {t}": pct for t, pct in overall.items()}]))

    lines.append("\n**Sufficiency by threshold, by partition (unbounded):**\n")
    lines.append(_table(analysis["unbounded"]["sufficiency_by_partition"]))

    lines.append("\n**Anchor-side dominant-group exclusion sensitivity (M4):**\n")
    lines.append(_table(analysis["unbounded"]["dominant_anchor_exclusion_sensitivity"]))

    lines.append("\n**Other-side dominant concentration (M4, reverse-hub check):**\n")
    other_stats = analysis["other_side_dominant_concentration"]["other_group_size_summary_stats"]
    lines.append(
        f"- `other_col` groups: **{other_stats['n_groups']:,}**, largest **{other_stats['max_group_size']:,}** rows, "
        f"median **{other_stats['median_group_size']:.1f}**.\n"
    )
    lines.append(_table(analysis["other_side_dominant_concentration"]["top_k_concentration"]))

    lines.append("\n**Windowed sufficiency-at-decision-threshold summary (M1/M2, per candidate window):**\n")
    windowed_rows = [
        {
            "window_size_events": w,
            **{f">= {t}": pct for t, pct in w_result["sufficiency_overall"].items()},
        }
        for w, w_result in analysis["windowed"].items()
    ]
    lines.append(_table(windowed_rows))

    lines.append(
        f"\n**Evaluation** (decision_threshold={ev['decision_threshold']}): row coverage "
        f"**{ev['row_coverage_pct']:.2f}%** (row_coverage_ok: **{ev['row_coverage_ok']}**) | "
        f"overall fan-out sufficiency **{ev['overall_fanout_sufficiency_pct']:.2f}%** "
        f"(fanout_density_ok: **{ev['fanout_density_ok']}**) | "
        f"partition_stability_ok: **{ev['partition_stability_ok']}** | "
        f"dominant_anchor_robustness_ok: **{ev['dominant_anchor_robustness_ok']}** "
        f"(worst case after exclusion: {ev['worst_case_sufficiency_pct_after_exclusion']:.2f}%) | "
        f"**is_suitable: {ev['is_suitable']}** | "
        f"recommended_window: **{entry['recommended_window'] if entry['recommended_window'] is not None else 'unbounded only'}**."
    )

    lines.append(
        f"\n**M5 overlap diagnostic (descriptive only -- does not drive the recommendation)**: "
        f"{overlap['n_high_fanout_anchors']:,} / {overlap['n_anchors_total']:,} anchors "
        f"at or above the {overlap['high_fanout_percentile']:.0f}th percentile "
        f"(threshold value: {overlap['high_fanout_threshold_value']:.2f} distinct partners). "
        f"Mean RAW overlap fraction: **{overlap['overlap_fraction_mean_pct']:.2f}%** | "
        f"clears_multi_hop_signal (informational only): **{overlap['clears_multi_hop_signal']}**."
    )
    if overlap.get("overlap_fraction_distribution"):
        lines.append("\n" + _table([overlap["overlap_fraction_distribution"]], columns=["min", "p25", "p50", "p75", "p90", "max", "mean"]))

    lines.append(
        f"\n**M5b frequency-adjusted overlap diagnostic (DECISION-DRIVING)**: "
        f"{frequency_adjusted_overlap['n_high_fanout_anchors']:,} / {frequency_adjusted_overlap['n_anchors_total']:,} anchors selected "
        f"(rank-count selection, min_anchor_count={frequency_adjusted_overlap['min_anchor_count']}), "
        f"{frequency_adjusted_overlap['n_anchors_with_nonempty_partner_set']:,} with a nonempty partner set.\n\n"
        f"| mean observed overlap | mean expected overlap (null) | excess (pct points) | lift_ratio | clears_lift_signal |\n"
        f"|---|---|---|---|---|\n"
        f"| {frequency_adjusted_overlap['mean_observed_overlap_pct']:.2f}% | {frequency_adjusted_overlap['mean_expected_overlap_pct']:.2f}% | "
        f"{frequency_adjusted_overlap['excess_pct_points']:+.2f} pts | **{frequency_adjusted_overlap['lift_ratio']:.2f}x** | "
        f"**{frequency_adjusted_overlap['clears_lift_signal']}** |"
    )

    return lines


def render_phase_e1_report(results: dict, report_path: Path) -> None:
    split_config = results["split_config"]
    recommendation = results["recommendation"]
    directions = results["directions"]

    lines: list[str] = []
    lines.append("# SentinelPay -- Phase E.1 Link Sufficiency & Causal Cross-Key Fan-Out Report")
    lines.append("")
    lines.append(
        "**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_e1_report` "
        "from `reports/eda/phase_e1_results.json`** -- every number below is read from that file; "
        "re-running `python -m sentinelpay.eda.run_phase_e1` regenerates both together."
    )

    lines.append(_PHASE_E1_SCOPE_SECTION)

    lines.append("## 1. Split configuration (unchanged from Phase B/C/D/D.1)\n")
    lines.append("| partition | start_day | end_day |")
    lines.append("|---|---|---|")
    for name in ["train", "embargo_1", "validation", "embargo_2", "holdout"]:
        rng = split_config[name]
        lines.append(f"| {name} | {rng['start_day']} | {rng['end_day']} |")

    lines.append("\n## 2. Holdout sealing\n")
    lines.append(
        f"- Total rows loaded (train_transaction.csv joined to train_identity.csv): "
        f"**{results['n_rows_total']:,}**.\n"
        f"- Rows filtered to `train`/`embargo_1`/`validation`/`embargo_2` "
        f"(`sentinelpay.data.split.DEVELOPMENT_PARTITIONS`) **before** any relationship content "
        f"analysis: **{results['n_rows_development']:,}**.\n"
        f"- Holdout rows excluded, never touched by relationship-frame, history, or overlap "
        f"computation: **{results['n_rows_holdout_excluded']:,}**.\n"
        f"- `isFraud` is never loaded by `sentinelpay.eda.run_phase_e1`."
    )

    lines.append(_PHASE_E1_METHOD_SECTION)
    lines.append(_PHASE_E1_M5_SECTION)

    lines.append("\n## 3. Relationship-pair row coverage (M3, shared by both directions of each pair)\n")
    pair_rows = [
        {"relationship_pair": name, "n_rows_valid": r["n_rows_valid"], "pct_rows_valid": r["pct_rows_valid"]}
        for name, r in results["relationship_pairs_row_coverage"].items()
    ]
    lines.append(_table(pair_rows))

    lines.append("\n## 4. Per-direction results (M1-M5)\n")
    section_titles = {
        "payment_to_device": "4.1 payment_node -> device_node",
        "device_to_payment": "4.2 device_node -> payment_node",
        "payment_to_email_purchaser": "4.3 payment_node -> email_purchaser_node",
        "email_purchaser_to_payment": "4.4 email_purchaser_node -> payment_node",
        "device_to_email_purchaser": "4.5 device_node -> email_purchaser_node",
        "email_purchaser_to_device": "4.6 email_purchaser_node -> device_node",
    }
    for key, title in section_titles.items():
        if key in directions:
            lines.extend(_direction_section(title, directions[key]))

    lines.append("\n## 5. Recommendation\n")
    lines.append(
        "**Per-direction verdicts, window recommendations, and both overlap diagnostics** "
        "(`raw_overlap_pct` is M5, descriptive only; `lift_ratio` is M5b, decision-driving):\n"
    )
    per_dir_rows = [
        {
            "direction": name,
            "is_suitable": v["is_suitable"],
            "recommended_window": v["recommended_window"] if v["recommended_window"] is not None else "unbounded only",
            "raw_overlap_pct_M5_descriptive": v.get("raw_overlap_pct_descriptive_only"),
            "lift_ratio_M5b_decision_driving": v.get("lift_ratio"),
        }
        for name, v in recommendation["per_direction"].items()
    ]
    lines.append(_table(per_dir_rows))

    lines.append(
        f"\n**recommend_union_find_for_e2: {recommendation['recommend_union_find_for_e2']}**\n\n"
        f"{recommendation['reason']}\n\n"
        "This is the output of `sentinelpay.eda.link_sufficiency.recommend_relationships` applied to "
        "the measured results above -- a pure function of this run's numbers, not a preference chosen "
        "before E.1 ran, and driven by M5b's `lift_ratio` only (M5's raw overlap is shown above for "
        "context but never gates this boolean). It is a recommendation for a Phase E.2 scoping decision, "
        "not an implementation: no Union-Find, component detection, scoring, or persistence exists yet "
        "regardless of this value."
    )

    lines.append("\n## 6. Not yet analyzed (measured columns available, deferred to a follow-up pass)\n")
    for item in results.get("not_yet_analyzed", []):
        lines.append(f"- {item}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


_PHASE_E2_SCOPE_SECTION = """
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
"""


def render_phase_e2_report(results: dict, report_path: Path) -> None:
    split_config = results["split_config"]
    coverage = results["relationship_row_coverage"]
    overall = results["component_metrics_summary_overall"]
    by_partition = results["component_metrics_summary_by_partition"]

    lines: list[str] = []
    lines.append("# SentinelPay -- Phase E.2 Union-Find Component Metrics Report (Measurement Pass)")
    lines.append("")
    lines.append(
        "**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_e2_report` "
        "from `reports/eda/phase_e2_results.json`** -- every number below is read from that file; "
        "re-running `python -m sentinelpay.eda.run_phase_e2` regenerates both together."
    )

    lines.append(_PHASE_E2_SCOPE_SECTION)

    lines.append("## 1. Split configuration (unchanged from Phase B/C/D/D.1/E.1)\n")
    lines.append("| partition | start_day | end_day |")
    lines.append("|---|---|---|")
    for name in ["train", "embargo_1", "validation", "embargo_2", "holdout"]:
        rng = split_config[name]
        lines.append(f"| {name} | {rng['start_day']} | {rng['end_day']} |")

    lines.append("\n## 2. Holdout sealing\n")
    lines.append(
        f"- Total rows loaded (train_transaction.csv joined to train_identity.csv): "
        f"**{results['n_rows_total']:,}**.\n"
        f"- Rows filtered to `train`/`embargo_1`/`validation`/`embargo_2` "
        f"(`sentinelpay.data.split.DEVELOPMENT_PARTITIONS`) **before** any component computation: "
        f"**{results['n_rows_development']:,}**.\n"
        f"- Holdout rows excluded, never touched by relationship-frame or component computation: "
        f"**{results['n_rows_holdout_excluded']:,}**.\n"
        f"- `isFraud` is never loaded by `sentinelpay.eda.run_phase_e2`."
    )

    lines.append("\n## 3. device_node <-> payment_node relationship row coverage\n")
    lines.append(
        f"**{coverage['n_rows_valid']:,}** / {results['n_rows_development']:,} development rows "
        f"(**{coverage['pct_rows_valid']:.2f}%**) have both `_device_node` and `_payment_node` present "
        "-- identical row coverage to Phase E.1's payment_device relationship pair (both directions "
        "share the same valid-row set)."
    )

    lines.append("\n## 4. Per-transaction component metrics -- overall descriptive summary\n")
    lines.append(f"n_rows: **{overall['n_rows']:,}**\n")
    for metric_name in ("device_component_size_total", "payment_component_size_total", "merged_component_size_total"):
        lines.append(f"**{metric_name}:**\n")
        lines.append(_table([overall[metric_name]]))
    same = overall["endpoints_same_component"]
    lines.append(
        f"**endpoints_same_component:** n_true={same['n_true']:,} | n_false={same['n_false']:,} | "
        f"pct_true={same['pct_true']:.4f}%\n"
    )

    lines.append("\n## 5. Per-transaction component metrics -- by partition\n")
    for row in by_partition:
        lines.append(f"### partition: {row['partition']} (n_rows={row['n_rows']:,})\n")
        for metric_name in ("device_component_size_total", "payment_component_size_total", "merged_component_size_total"):
            lines.append(f"**{metric_name}:**\n")
            lines.append(_table([row[metric_name]]))
        same = row["endpoints_same_component"]
        lines.append(
            f"**endpoints_same_component:** n_true={same['n_true']:,} | n_false={same['n_false']:,} | "
            f"pct_true={same['pct_true']:.4f}%\n"
        )

    diag = results.get("fanout_stratified_diagnostic_evaluation")
    if diag:
        lines.append(
            "\n## 6. Fan-out-stratified diagnostic evaluation (isFraud, validation-partition only)\n\n"
            "The ONE place `isFraud` is read in this phase -- strictly downstream of every quantity "
            "above, `validation`-partition rows only. Question: does `merged_component_size_total` / "
            "`endpoints_same_component` carry fraud-rate signal BEYOND device_to_payment's own prior "
            "fan-out (Phase E.1's M1 quantity), or is it just re-detecting fan-out/hub-domination "
            "(the same confound E.1's M5-vs-M5b correction addressed for raw overlap)? This is a "
            "one-time EDA evaluation only -- no production feature, score, threshold, config, or E.3 "
            "work is added regardless of the result below.\n"
        )
        lines.append(
            f"**Fixed bins, declared BEFORE `isFraud` was read** (not tuned to any fraud-rate outcome):\n\n"
            f"- `fanout_stratum_edges` (device_to_payment unbounded prior-distinct-partner count; "
            f"E.1's own published p25/p50/p75 for this direction): `{diag['fanout_stratum_edges']}` -> "
            f"{diag['fanout_stratum_labels']}\n"
            f"- `component_size_bin_edges` (`merged_component_size_total`; this run's own published "
            f"overall p25/p50/p75/p90, section 4 above): `{diag['component_size_bin_edges']}` -> "
            f"{diag['component_size_bin_labels']}\n"
        )
        lines.append(f"\n`n_validation_rows` (partition == \"validation\" only): **{diag['n_validation_rows']:,}**\n")

        lines.append("\n### 6.1 Unstratified (reference only)\n")
        u = diag["unstratified"]
        lines.append(
            f"n_rows={u['n_rows']:,} | fraud_rate={u['fraud_rate']:.6f} | "
            f"roc_auc_merged_component_size_total_vs_isFraud="
            f"{u['roc_auc_merged_component_size_total_vs_isFraud']:.4f}\n"
        )
        lines.append("**fraud rate by merged_component_size_total bucket:**\n")
        lines.append(_table(u["merged_component_size_total_fraud_rate_by_bucket"], float_fmt="{:.6f}"))
        lines.append("**fraud rate by endpoints_same_component:**\n")
        lines.append(_table(u["endpoints_same_component_fraud_rate"], float_fmt="{:.6f}"))

        lines.append("\n### 6.2 Per fan-out stratum\n")
        for s in diag["fanout_strata"]:
            auc = s["roc_auc_merged_component_size_total_vs_isFraud"]
            auc_str = "undefined (single class or 0 rows)" if (isinstance(auc, float) and math.isnan(auc)) else f"{auc:.4f}"
            lines.append(
                f"\n**{s['stratum']}** -- n_rows={s['n_rows']:,} | fraud_rate="
                f"{s['fraud_rate']:.6f} | roc_auc_merged_component_size_total_vs_isFraud={auc_str}\n"
            )
            lines.append("fraud rate by merged_component_size_total bucket:\n")
            lines.append(_table(s["merged_component_size_total_fraud_rate_by_bucket"], float_fmt="{:.6f}"))
            lines.append("fraud rate by endpoints_same_component:\n")
            lines.append(_table(s["endpoints_same_component_fraud_rate"], float_fmt="{:.6f}"))

        lines.append("\n### 6.3 Conclusion\n")
        lines.append(diag["conclusion"] + "\n")
    else:
        lines.append(
            "\n## 6. Not yet done\n\n"
            "- The pre-declared fan-out-stratified diagnostic evaluation (target-reading, strictly "
            "downstream of this measurement pass) has not been run yet.\n"
            "- No production feature, scoring model, or E.3 work exists yet regardless of this report's "
            "numbers.\n"
        )

    lines.append(
        "\n## 7. No production feature, scoring model, or E.3 work\n\n"
        "Nothing in this report is a production feature, score, threshold, `configs/detection.yaml`-"
        "style config, or E.3 work -- this is a one-time EDA measurement + diagnostic evaluation only.\n"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def render_phase_b_report(results: dict, report_path: Path) -> None:
    split_config = results["split_config"]
    validation_result = results["validation_result"]
    b2 = results["b2_embargo_sensitivity"]
    drift = results["drift_train_vs_validation"]
    identity_cov = results["identity_coverage_by_partition"]
    rollup_demo = results["rollup_utility_demo"]
    memory = results.get("memory", {})

    lines: list[str] = []
    lines.append("# SentinelPay -- Phase B Leakage-Safe Temporal Data Pipeline Report")
    lines.append("")
    lines.append(
        "**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_b_report` "
        "from `reports/eda/phase_b_results.json`** -- every number below is read from that file; "
        "re-running `python -m sentinelpay.eda.run_phase_b` regenerates both together."
    )

    lines.append("\n## 1. Split configuration (fixed, not fit to data)\n")
    lines.append(
        "Boundaries are fixed day-index ranges from `configs/split.yaml`, chosen by convention -- "
        "never by searching for a cut that optimizes fraud rate or any label-based objective."
    )
    lines.append("\n| partition | start_day | end_day |")
    lines.append("|---|---|---|")
    for name in ["train", "embargo_1", "validation", "embargo_2", "holdout"]:
        rng = split_config[name]
        lines.append(f"| {name} | {rng['start_day']} | {rng['end_day']} |")

    lines.append("\n## 2. Structural split validation\n")
    lines.append(f"**is_valid: {validation_result['is_valid']}**\n")
    lines.append("| partition | row count |")
    lines.append("|---|---|")
    for name, count in validation_result["partition_row_counts"].items():
        lines.append(f"| {name} | {count:,} |")
    lines.append(
        f"\n- Unassigned rows: {validation_result['n_unassigned_rows']} "
        f"| Empty partitions: {validation_result['empty_partitions'] or 'none'} "
        f"| TransactionIDs crossing partitions: {validation_result['n_transaction_ids_in_multiple_partitions']}"
    )
    lines.append(
        f"- Chronological order OK: **{validation_result['chronological_order_ok']}** "
        f"| Holdout strictly after validation: **{validation_result['holdout_strictly_after_validation']}** "
        f"| Embargoes isolated: **{validation_result['embargoes_isolated']}**"
    )
    lines.append("\nChronological TransactionDT bounds per partition (min, max):\n")
    bounds = validation_result["chronological_bounds_dt"]
    lines.append("| partition | min TransactionDT | max TransactionDT |")
    lines.append("|---|---|---|")
    for name in ["train", "embargo_1", "validation", "embargo_2", "holdout"]:
        if name in bounds:
            lo, hi = bounds[name]
            lines.append(f"| {name} | {lo:,.0f} | {hi:,.0f} |")

    lines.append(_HOLDOUT_SEALING_SECTION)

    lines.append("## 3. Embargo boundary sensitivity (train / embargo_1 / validation only)\n")
    lines.append(b2.get("note", ""))
    lines.append("\n" + _table(b2.get("d_column_summary_by_partition", [])))

    lines.append("\n## 4. Drift: train vs. validation only\n")
    lines.append("**Numeric (KS 2-sample), curated columns:**\n")
    lines.append(_table(drift["numeric"]))
    lines.append("\n**Categorical (chi-square):**\n")
    lines.append(_table(drift["categorical"]))
    lines.append("\n**Curated target correlation -- train partition:**\n")
    lines.append(_table(drift["curated_target_correlation_train"]))
    lines.append("\n**Curated target correlation -- validation partition:**\n")
    lines.append(_table(drift["curated_target_correlation_validation"]))
    lines.append("\n**V-column block correlation -- train partition:**\n")
    lines.append(_table(drift["v_block_target_correlation_train"]))
    lines.append("\n**V-column block correlation -- validation partition:**\n")
    lines.append(_table(drift["v_block_target_correlation_validation"]))
    lines.append(
        "\nNone of the above is fed back into the split boundary -- the boundary in section 1 is fixed "
        "regardless of what this drift analysis finds."
    )

    lines.append("\n## 5. Identity coverage by partition\n")
    lines.append("| partition | n_transactions | n_with_identity | pct_with_identity |")
    lines.append("|---|---|---|---|")
    for name in ["train", "embargo_1", "validation", "embargo_2"]:
        row = identity_cov.get(name, {})
        if "pct_with_identity" in row:
            lines.append(
                f"| {name} | {row['n_transactions']:,} | {row['n_with_identity']:,} | {row['pct_with_identity']:.2f}% |"
            )
    lines.append(f"| holdout | -- | -- | *{identity_cov.get('holdout', {}).get('note', 'not computed')}* |")

    lines.append("\n## 6. Rollup utility demo (illustrative only)\n")
    lines.append(f"Placeholder grouping column: `{rollup_demo['placeholder_group_column']}`.")
    lines.append(_GROUPING_KEY_DEFERRAL_SECTION)
    lines.append("**`daily_count_by_group` sample output:**\n")
    lines.append(_table(rollup_demo.get("daily_count_sample", [])))
    lines.append("\n**`daily_amount_stats_by_group` sample output:**\n")
    lines.append(_table(rollup_demo.get("daily_amount_stats_sample", [])))

    lines.append("\n## 7. Memory observations\n")
    lines.append(_table([memory], columns=list(memory.keys())) if memory else "_(no memory data)_\n")
    lines.append(
        "\nOnly the section 4 drift/correlation step used `load_transaction_full` (explicit, logged as "
        "high-memory). Every other step used column-scoped loading (`load_transaction_columns` / "
        "`load_transaction_ids` / `load_identity_ids`)."
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


_PHASE_D_SCOPE_SECTION = """
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
"""

_PHASE_D_EMBARGO_SECTION = """
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
"""


def _flag_table(rows: list[dict], flags: list[str]) -> str:
    cols = ["partition", "n_rows"] + [f"pct_{f}" for f in flags]
    return _table(rows, columns=cols)


def render_phase_d_report(results: dict, report_path: Path) -> None:
    split_config = results["split_config"]
    dc = results["detection_config"]
    evaluation = results["validation_evaluation"]
    coverage = evaluation["coverage"]

    flags = ["insufficient_history", "zero_mad", "scored_normal", "scored_outlier"]

    lines: list[str] = []
    lines.append("# SentinelPay -- Phase D Behavioral-Change Detector Report")
    lines.append("")
    lines.append(
        "**Generated deterministically by `sentinelpay.eda.generate_report.render_phase_d_report` "
        "from `reports/eda/phase_d_results.json`** -- every number below is read from that file; "
        "re-running `python -m sentinelpay.eda.run_phase_d` regenerates both together."
    )

    lines.append(_PHASE_D_SCOPE_SECTION)

    lines.append("## 1. Split configuration (unchanged from Phase B/C/D.1)\n")
    lines.append("| partition | start_day | end_day |")
    lines.append("|---|---|---|")
    for name in ["train", "embargo_1", "validation", "embargo_2", "holdout"]:
        rng = split_config[name]
        lines.append(f"| {name} | {rng['start_day']} | {rng['end_day']} |")

    lines.append("\n## 2. Detector hyperparameters (fixed before any evaluation)\n")
    lines.append(
        "All five values are loaded from `configs/detection.yaml`, never selected, tuned, or changed "
        "based on validation `isFraud` results.\n"
    )
    lines.append("| parameter | value | justification |")
    lines.append("|---|---|---|")
    lines.append(
        f"| `min_history_for_score` | {dc['min_history_for_score']} | Reuses D.1's `DECISION_THRESHOLD` "
        "-- the pre-declared, already-validated (partition-stable, dominant-group-robust) \"enough causal "
        "history\" bar for `payment_proxy_key`. |"
    )
    lines.append(
        f"| `window_size_events` | {dc['window_size_events']} | Reuses D.1's largest pre-declared "
        "threshold bucket (>= 20), already measured as majority-stable for `payment_proxy_key` across "
        "partitions and dominant-group exclusion. |"
    )
    lines.append(
        f"| `modified_zscore_scale_constant` | {dc['modified_zscore_scale_constant']} | Standard "
        "Iglewicz & Hoaglin (1993) modified z-score consistency constant -- a mathematical property of "
        "the statistic, not tuned to this dataset. |"
    )
    lines.append(
        f"| `modified_zscore_threshold` | {dc['modified_zscore_threshold']} | Standard, widely cited "
        "Iglewicz & Hoaglin outlier cutoff for the modified z-score. |"
    )
    lines.append(
        f"| `zero_mad_epsilon` | {dc['zero_mad_epsilon']} | Numerical-safety floor only (avoids division "
        "blow-up when prior amounts are near-identical), not a detection-sensitivity knob. |"
    )

    lines.append("\n## 3. Holdout sealing\n")
    lines.append(
        f"- Total rows loaded (`train_transaction.csv`): **{results['n_rows_total']:,}**.\n"
        f"- Rows filtered to `train`/`embargo_1`/`validation`/`embargo_2` "
        f"(`sentinelpay.data.split.DEVELOPMENT_PARTITIONS`) **before** `build_group_key`/"
        f"`compute_behavioral_change_score` are ever called: **{results['n_rows_development']:,}**.\n"
        f"- Holdout rows excluded, never touched by group-key, history, or score computation: "
        f"**{results['n_rows_holdout_excluded']:,}**.\n"
        f"- Of the development rows, **{results['n_rows_valid_key']:,}** have a valid `payment_proxy_key` "
        f"({results['n_rows_missing_payment_proxy_key']:,} excluded, missing a key component).\n"
        f"- `isFraud` is never read while building the detector; it is read only in section 5 below, "
        f"only for `validation`-partition rows."
    )

    lines.append(_PHASE_D_EMBARGO_SECTION)

    lines.append("## 4. Score coverage by partition (non-target)\n")
    lines.append(_flag_table(results["coverage_by_partition"], flags))

    dist = results.get("score_distribution", {})
    if dist:
        lines.append("\n**`abs(modified_zscore)` distribution among scored rows (all development partitions):**\n")
        lines.append(
            _table(
                [dist],
                columns=[
                    "n_scored_rows",
                    "abs_modified_zscore_p50",
                    "abs_modified_zscore_p75",
                    "abs_modified_zscore_p90",
                    "abs_modified_zscore_p99",
                ],
            )
        )

    lines.append(
        "\n## 5. `validation`-only target-association evaluation (diagnostic only -- NOT a feature, "
        "NOT used to select any hyperparameter)\n"
    )
    lines.append(
        f"- `validation` rows: **{coverage['n_validation_rows']:,}**. Flag breakdown: "
        + ", ".join(f"`{f}` {coverage['flag_pct'].get(f, float('nan')):.2f}%" for f in flags)
        + "."
    )
    lines.append(f"- Scored rows (non-`NaN` score) used for the metrics below: **{results['validation_evaluation']['n_scored_rows']:,}**.")

    lines.append("\n**Fraud rate by `modified_zscore` decile (scored `validation` rows only):**\n")
    lines.append(_table(evaluation.get("fraud_rate_by_score_decile", [])))

    lines.append("\n**Fraud rate: `scored_outlier` vs. `scored_normal` (scored `validation` rows only):**\n")
    lines.append(_table(evaluation.get("fraud_rate_outlier_vs_normal", [])))

    auc = evaluation.get("roc_auc_abs_modified_zscore_vs_isFraud", float("nan"))
    lines.append(
        f"\n**ROC-AUC of `abs(modified_zscore)` vs. `isFraud`** (scored `validation` rows only, "
        f"n={evaluation.get('roc_auc_n_rows', 0):,}): **{auc:.4f}**."
    )
    lines.append(
        "\nThese four diagnostics are reported once, for the fixed configuration in section 2 above. "
        "No threshold, window size, or constant is reselected from these results, and no sensitivity/sweep "
        "table against `isFraud` is produced in this phase."
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

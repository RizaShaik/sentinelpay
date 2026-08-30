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

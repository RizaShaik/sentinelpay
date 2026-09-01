import json

import pandas as pd
import pytest

from sentinelpay.eda.grouping_key_sufficiency import (
    analyze_grouping_key,
    build_group_key,
    causal_prior_counts,
    dominant_group_exclusion_sensitivity,
    evaluate_key_sufficiency,
    prior_count_distribution,
    recommend_grouping_key,
    sufficiency_at_thresholds,
    sufficiency_by_partition,
)


def _same_result(a: dict, b: dict) -> bool:
    # Plain `==` treats NaN != NaN, which real sufficiency results can
    # contain (e.g. a threshold with zero remaining rows). Compare via a
    # deterministic string dump instead, where NaN renders identically on
    # both sides.
    dump = lambda d: json.dumps(d, sort_keys=True, default=str)
    return dump(a) == dump(b)


def _synthetic_dev_df():
    # Deliberately synthetic key columns -- never a real proxy key. Rows 2
    # and 3 share group "A" and the same TransactionDT (200) for tie
    # handling. Row 6 has a missing key component (invalid for grouping).
    return pd.DataFrame(
        {
            "keyA": ["A", "A", "A", "A", "B", "A", None],
            "keyB": [1, 1, 1, 1, 2, 1, 3],
            "TransactionDT": [100, 150, 200, 200, 100, 300, 50],
            "partition": [
                "train", "train", "train", "embargo_1", "train", "validation", "train",
            ],
        }
    )


def test_build_group_key_drops_missing_and_combines_columns():
    df = _synthetic_dev_df()
    out = build_group_key(df, ["keyA", "keyB"], key_name="_gk")
    assert len(out) == 6  # last row (missing keyA) dropped
    assert out["_gk"].nunique() == 2  # group A (keyA=A,keyB=1) and group B (keyA=B,keyB=2)


def test_build_group_key_requires_columns():
    df = _synthetic_dev_df()
    with pytest.raises(ValueError):
        build_group_key(df, ["no_such_col"])


def test_causal_prior_counts_matches_history_semantics():
    df = build_group_key(_synthetic_dev_df(), ["keyA", "keyB"], key_name="_gk")
    counts = causal_prior_counts(df, group_col="_gk", dt_col="TransactionDT")
    by_dt = {dt: c for dt, c in zip(df["TransactionDT"], counts)}
    # group A sorted by dt: 100, 150, 200, 200, 300
    assert by_dt[100] == 0
    assert by_dt[150] == 1
    assert by_dt[300] == 4


def test_tied_timestamps_never_see_each_other():
    df = build_group_key(_synthetic_dev_df(), ["keyA", "keyB"], key_name="_gk")
    counts = causal_prior_counts(df, group_col="_gk", dt_col="TransactionDT")
    tied = df[df["TransactionDT"] == 200]
    tied_counts = counts.loc[tied.index]
    assert (tied_counts == 2).all()  # both tied rows see only the two dt<200 rows, never each other


def test_future_perturbation_does_not_change_earlier_rows():
    df = build_group_key(_synthetic_dev_df(), ["keyA", "keyB"], key_name="_gk")
    before = causal_prior_counts(df, group_col="_gk", dt_col="TransactionDT")

    df_mutated = df.copy()
    last_idx = df_mutated["TransactionDT"].idxmax()
    df_mutated.loc[last_idx, "TransactionDT"] = 999_999
    after = causal_prior_counts(df_mutated, group_col="_gk", dt_col="TransactionDT")

    earlier_idx = [i for i in df.index if i != last_idx]
    for idx in earlier_idx:
        assert before.loc[idx] == after.loc[idx]


def test_row_order_independence():
    df = build_group_key(_synthetic_dev_df(), ["keyA", "keyB"], key_name="_gk")
    shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)

    counts_orig = causal_prior_counts(df, group_col="_gk", dt_col="TransactionDT")
    counts_shuf = causal_prior_counts(shuffled, group_col="_gk", dt_col="TransactionDT")

    lookup = {(g, dt): c for g, dt, c in zip(df["_gk"], df["TransactionDT"], counts_orig)}
    for g, dt, c in zip(shuffled["_gk"], shuffled["TransactionDT"], counts_shuf):
        assert lookup[(g, dt)] == c


def test_sufficiency_at_thresholds_basic():
    counts = pd.Series([0, 1, 2, 5, 10, 20, 20])
    result = sufficiency_at_thresholds(counts, thresholds=[1, 5, 10, 20])
    n = 7
    assert result[1] == pytest.approx(100.0 * 6 / n, abs=1e-3)  # all but the 0
    assert result[5] == pytest.approx(100.0 * 4 / n, abs=1e-3)  # 5,10,20,20
    assert result[20] == pytest.approx(100.0 * 2 / n, abs=1e-3)


def test_sufficiency_at_thresholds_empty_series_is_nan():
    result = sufficiency_at_thresholds(pd.Series([], dtype="int64"), thresholds=[1, 5])
    assert all(pd.isna(v) for v in result.values())


def test_prior_count_distribution_scalars():
    counts = pd.Series([0, 1, 2, 3, 4, 5, 100])
    dist = prior_count_distribution(counts)
    assert dist["min"] == 0
    assert dist["max"] == 100
    assert dist["mean"] == pytest.approx(counts.mean())


def test_sufficiency_by_partition_breaks_out_correctly():
    df = build_group_key(_synthetic_dev_df(), ["keyA", "keyB"], key_name="_gk")
    counts = causal_prior_counts(df, group_col="_gk", dt_col="TransactionDT")
    rows = sufficiency_by_partition(df, "partition", counts, ["train", "embargo_1", "validation"], thresholds=[1])
    by_partition = {r["partition"]: r for r in rows}
    assert by_partition["train"]["n_valid_rows"] == 4  # dt 100,150,200 (group A) + dt 100 (group B)
    assert by_partition["embargo_1"]["n_valid_rows"] == 1
    assert by_partition["validation"]["n_valid_rows"] == 1


def test_sufficiency_by_partition_requires_column():
    df = build_group_key(_synthetic_dev_df(), ["keyA", "keyB"], key_name="_gk")
    counts = causal_prior_counts(df, group_col="_gk", dt_col="TransactionDT")
    with pytest.raises(ValueError):
        sufficiency_by_partition(df, "no_such_partition_col", counts, ["train"])


def test_dominant_group_exclusion_sensitivity_reports_shrinking_denominator():
    # 20 rows in dominant group "D", 5 rows spread across small groups.
    dominant_rows = pd.DataFrame({"_gk": ["D"] * 20, "TransactionDT": list(range(20))})
    small_rows = pd.DataFrame({"_gk": [f"S{i}" for i in range(5)], "TransactionDT": list(range(20, 25))})
    df = pd.concat([dominant_rows, small_rows], ignore_index=True)
    counts = causal_prior_counts(df, group_col="_gk", dt_col="TransactionDT")

    result = dominant_group_exclusion_sensitivity(df, "_gk", counts, top_k_list=[1], n_development_total=25, thresholds=[1])
    row = result[0]
    assert row["excluded_group_sizes"] == [20]
    assert row["n_valid_rows_remaining"] == 5
    assert row["pct_valid_rows_remaining_of_original_valid"] == pytest.approx(100.0 * 5 / 25)
    assert row["pct_valid_rows_remaining_of_development_total"] == pytest.approx(100.0 * 5 / 25)
    # every small-group row is a singleton (no prior events) -> 0% sufficient at threshold 1
    assert row["pct_sufficient_ge_1"] == pytest.approx(0.0)


def test_dominant_group_exclusion_sensitivity_requires_column():
    df = pd.DataFrame({"_gk": ["A"], "TransactionDT": [1]})
    counts = pd.Series([0])
    with pytest.raises(ValueError):
        dominant_group_exclusion_sensitivity(df, "no_such_col", counts, top_k_list=[1])


def test_analyze_grouping_key_end_to_end_structure():
    df = _synthetic_dev_df()
    result = analyze_grouping_key(
        df,
        key_columns=["keyA", "keyB"],
        dt_col="TransactionDT",
        partition_col="partition",
        partitions=["train", "embargo_1", "validation"],
        thresholds=[1, 3],
        top_k_list=[1],
    )
    assert result["n_rows_development"] == 7
    assert result["n_rows_valid"] == 6
    assert result["n_groups"] == 2
    assert result["n_singleton_groups"] == 1  # group B
    assert set(result["sufficiency_overall"].keys()) == {1, 3}
    assert len(result["sufficiency_by_partition"]) == 3
    assert len(result["dominant_group_exclusion_sensitivity"]) == 1


def test_holdout_rows_never_influence_analysis_when_excluded_before_the_call():
    df = _synthetic_dev_df()
    non_holdout = df[df["partition"] != "holdout"].copy()  # already excludes nothing here; add real holdout rows below

    # Add holdout rows that would, if included, massively change group A's history.
    holdout_rows = pd.DataFrame(
        {
            "keyA": ["A"] * 50,
            "keyB": [1] * 50,
            "TransactionDT": list(range(1000, 1050)),
            "partition": ["holdout"] * 50,
        }
    )
    df_with_holdout = pd.concat([df, holdout_rows], ignore_index=True)
    development_only = df_with_holdout[df_with_holdout["partition"] != "holdout"].copy()

    result_without_holdout_ever_present = analyze_grouping_key(
        non_holdout, ["keyA", "keyB"], "TransactionDT", "partition", ["train", "embargo_1", "validation"]
    )
    result_after_filtering_holdout_out = analyze_grouping_key(
        development_only, ["keyA", "keyB"], "TransactionDT", "partition", ["train", "embargo_1", "validation"]
    )

    assert _same_result(result_without_holdout_ever_present, result_after_filtering_holdout_out)


def test_no_target_dependency_behavioral():
    df = _synthetic_dev_df()
    df_with_target = df.copy()
    df_with_target["isFraud"] = [1, 0, 1, 0, 1, 0, 1]
    df_shuffled_target = df.copy()
    df_shuffled_target["isFraud"] = [0, 1, 0, 1, 0, 1, 0]

    kwargs = dict(
        key_columns=["keyA", "keyB"],
        dt_col="TransactionDT",
        partition_col="partition",
        partitions=["train", "embargo_1", "validation"],
    )
    result_a = analyze_grouping_key(df, **kwargs)
    result_b = analyze_grouping_key(df_with_target, **kwargs)
    result_c = analyze_grouping_key(df_shuffled_target, **kwargs)

    assert _same_result(result_a, result_b)
    assert _same_result(result_a, result_c)


def _good_result(overall_pct=80.0, partition_pcts=None, dominant_pcts=None, pct_rows_valid=90.0):
    partition_pcts = partition_pcts or {"train": 80.0, "embargo_1": 78.0, "validation": 75.0}
    dominant_pcts = dominant_pcts if dominant_pcts is not None else [78.0, 76.0, 70.0, 65.0]
    return {
        "pct_rows_valid": pct_rows_valid,
        "sufficiency_overall": {5: overall_pct},
        "sufficiency_by_partition": [{"partition": p, "pct_sufficient_ge_5": v} for p, v in partition_pcts.items()],
        "dominant_group_exclusion_sensitivity": [
            {"top_k_excluded": k, "pct_sufficient_ge_5": v} for k, v in zip([1, 3, 5, 10], dominant_pcts)
        ],
    }


def test_evaluate_key_sufficiency_passes_when_stable_and_robust():
    result = _good_result()
    ev = evaluate_key_sufficiency(result, threshold=5)
    assert ev["coverage_ok"]
    assert ev["partition_stability_ok"]
    assert ev["dominant_group_robustness_ok"]
    assert ev["is_suitable"]


def test_evaluate_key_sufficiency_fails_on_low_coverage():
    result = _good_result(overall_pct=10.0, partition_pcts={"train": 10.0, "embargo_1": 9.0, "validation": 8.0})
    ev = evaluate_key_sufficiency(result, threshold=5)
    assert not ev["coverage_ok"]
    assert not ev["is_suitable"]


def test_evaluate_key_sufficiency_fails_on_dominant_group_collapse():
    # Sufficiency collapses to near-zero once dominant groups are excluded.
    result = _good_result(dominant_pcts=[5.0, 3.0, 1.0, 0.5])
    ev = evaluate_key_sufficiency(result, threshold=5)
    assert not ev["dominant_group_robustness_ok"]
    assert not ev["is_suitable"]


def test_evaluate_key_sufficiency_fails_on_partition_concentration():
    # Sufficiency only holds in train, collapses in validation/embargo.
    result = _good_result(overall_pct=80.0, partition_pcts={"train": 80.0, "embargo_1": 5.0, "validation": 3.0})
    ev = evaluate_key_sufficiency(result, threshold=5)
    assert not ev["partition_stability_ok"]
    assert not ev["is_suitable"]


def test_evaluate_key_sufficiency_fails_on_low_row_coverage_even_if_density_is_high():
    # High density among the rows that have the key, but the key itself is
    # only present on a minority of transactions -- should not be suitable
    # as a population-wide per-entity key.
    result = _good_result(pct_rows_valid=20.0)
    ev = evaluate_key_sufficiency(result, threshold=5)
    assert not ev["row_coverage_ok"]
    assert not ev["is_suitable"]


def test_recommend_grouping_key_prefers_higher_sufficiency_when_both_suitable():
    payment_eval = {"is_suitable": True, "overall_sufficiency_pct": 90.0}
    device_eval = {"is_suitable": True, "overall_sufficiency_pct": 60.0}
    rec = recommend_grouping_key(payment_eval, device_eval)
    assert rec["recommendation"] == "payment_proxy_key"


def test_recommend_grouping_key_picks_the_only_suitable_one():
    payment_eval = {"is_suitable": False, "overall_sufficiency_pct": 40.0}
    device_eval = {"is_suitable": True, "overall_sufficiency_pct": 60.0}
    rec = recommend_grouping_key(payment_eval, device_eval)
    assert rec["recommendation"] == "device_proxy_key"


def test_recommend_grouping_key_neither_suitable_but_nonzero_coverage():
    payment_eval = {"is_suitable": False, "overall_sufficiency_pct": 40.0}
    device_eval = {"is_suitable": False, "overall_sufficiency_pct": 30.0}
    rec = recommend_grouping_key(payment_eval, device_eval)
    assert rec["recommendation"] == "neither"


def test_recommend_grouping_key_zero_coverage_suggests_different_strategy():
    payment_eval = {"is_suitable": False, "overall_sufficiency_pct": 0.0}
    device_eval = {"is_suitable": False, "overall_sufficiency_pct": 30.0}
    rec = recommend_grouping_key(payment_eval, device_eval)
    assert rec["recommendation"] == "different_strategy"

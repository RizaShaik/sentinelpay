import inspect
import json

import pandas as pd
import pytest

from sentinelpay.eda.link_sufficiency import (
    HIGH_FANOUT_MIN_ANCHOR_COUNT,
    LIFT_SIGNAL_THRESHOLD,
    analyze_relationship_direction,
    build_node_key_column,
    build_relationship_frame,
    compute_partner_prevalence,
    evaluate_relationship_sufficiency,
    frequency_adjusted_overlap_diagnostic,
    other_side_top_k_concentration,
    overlap_diagnostic,
    recommend_relationships,
    recommend_window_size,
    relationship_row_coverage,
    select_high_fanout_anchors,
)


def _same_result(a: dict, b: dict) -> bool:
    dump = lambda d: json.dumps(d, sort_keys=True, default=str)
    return dump(a) == dump(b)


# ---------------------------------------------------------------------------
# build_node_key_column / build_relationship_frame
# ---------------------------------------------------------------------------


def test_build_node_key_column_preserves_rows_and_marks_missing_as_nan():
    df = pd.DataFrame({"a": ["x", "y", None], "b": [1, 2, 3]})
    out = build_node_key_column(df, ["a", "b"])
    assert len(out) == 3  # no rows dropped, unlike build_group_key
    assert out.iloc[0] == "x|1"
    assert out.iloc[1] == "y|2"
    assert pd.isna(out.iloc[2])


def test_build_node_key_column_requires_columns():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        build_node_key_column(df, ["no_such_col"])


def test_build_relationship_frame_drops_missing_either_side():
    df = pd.DataFrame({"anchor": ["A", "B", None, "D"], "other": ["x", None, "z", "w"]})
    out = build_relationship_frame(df, "anchor", "other")
    assert len(out) == 2
    assert set(out.index) == {0, 3}


def test_build_relationship_frame_requires_columns():
    df = pd.DataFrame({"anchor": ["A"]})
    with pytest.raises(ValueError):
        build_relationship_frame(df, "anchor", "no_such_col")


# ---------------------------------------------------------------------------
# other_side_top_k_concentration
# ---------------------------------------------------------------------------


def test_other_side_top_k_concentration_basic():
    df = pd.DataFrame({"other": ["D"] * 5 + ["e1", "e2", "e3"]})
    rows = other_side_top_k_concentration(df, "other", top_k_list=[1, 3])
    by_k = {r["top_k"]: r for r in rows}
    assert by_k[1]["pct_pairs_touching_top_k_other_values"] == pytest.approx(100.0 * 5 / 8)
    assert by_k[3]["pct_pairs_touching_top_k_other_values"] == pytest.approx(100.0 * 7 / 8)


def test_other_side_top_k_concentration_requires_column():
    df = pd.DataFrame({"other": ["A"]})
    with pytest.raises(ValueError):
        other_side_top_k_concentration(df, "no_such_col", top_k_list=[1])


# ---------------------------------------------------------------------------
# analyze_relationship_direction / relationship_row_coverage
# ---------------------------------------------------------------------------


def _synthetic_relationship_df():
    # anchor A sorted by dt: 100(X), 150(Y), 200(X)/200(Z) tie, 300(Y).
    # anchor B: 100(P), only row.
    return pd.DataFrame(
        {
            "anchor": ["A", "A", "A", "A", "B", "A"],
            "other": ["X", "Y", "X", "Z", "P", "Y"],
            "TransactionDT": [100, 150, 200, 200, 100, 300],
            "partition": ["train", "train", "train", "embargo_1", "train", "validation"],
        }
    )


def test_analyze_relationship_direction_end_to_end_structure():
    df = _synthetic_relationship_df()
    result = analyze_relationship_direction(
        df,
        anchor_col="anchor",
        other_col="other",
        dt_col="TransactionDT",
        partition_col="partition",
        partitions=["train", "embargo_1", "validation"],
        thresholds=[1, 2],
        window_candidates=[2],
        top_k_list=[1],
    )
    assert result["n_rows_valid"] == 6
    assert result["n_anchor_groups"] == 2
    assert result["n_anchor_singleton_groups"] == 1  # group B
    assert set(result["unbounded"]["sufficiency_overall"].keys()) == {1, 2}
    assert len(result["unbounded"]["sufficiency_by_partition"]) == 3
    assert len(result["unbounded"]["dominant_anchor_exclusion_sensitivity"]) == 1
    assert set(result["windowed"].keys()) == {2}
    assert "other_group_size_summary_stats" in result["other_side_dominant_concentration"]
    assert len(result["other_side_dominant_concentration"]["top_k_concentration"]) == 1


def test_relationship_row_coverage_basic():
    df = _synthetic_relationship_df()
    result = relationship_row_coverage(
        df, n_development_total=10, partition_col="partition", partitions=["train", "embargo_1", "validation"]
    )
    assert result["n_rows_valid"] == 6
    assert result["pct_rows_valid"] == pytest.approx(60.0)
    by_p = {r["partition"]: r["n_valid_rows"] for r in result["by_partition"]}
    assert by_p["train"] == 4
    assert by_p["embargo_1"] == 1
    assert by_p["validation"] == 1


def test_holdout_rows_never_influence_analysis_when_excluded_before_the_call():
    df = _synthetic_relationship_df()
    holdout_rows = pd.DataFrame(
        {
            "anchor": ["A"] * 20,
            "other": [f"H{i}" for i in range(20)],
            "TransactionDT": list(range(2000, 2020)),
            "partition": ["holdout"] * 20,
        }
    )
    df_with_holdout = pd.concat([df, holdout_rows], ignore_index=True)
    development_only = df_with_holdout[df_with_holdout["partition"] != "holdout"].copy()

    kwargs = dict(
        anchor_col="anchor",
        other_col="other",
        dt_col="TransactionDT",
        partition_col="partition",
        partitions=["train", "embargo_1", "validation"],
    )
    result_without_holdout_ever_present = analyze_relationship_direction(df, **kwargs)
    result_after_filtering = analyze_relationship_direction(development_only, **kwargs)
    assert _same_result(result_without_holdout_ever_present, result_after_filtering)


def test_no_target_dependency_behavioral():
    df = _synthetic_relationship_df()
    df_with_target = df.copy()
    df_with_target["isFraud"] = [1, 0, 1, 0, 1, 0]
    df_shuffled_target = df.copy()
    df_shuffled_target["isFraud"] = [0, 1, 0, 1, 0, 1]

    kwargs = dict(
        anchor_col="anchor",
        other_col="other",
        dt_col="TransactionDT",
        partition_col="partition",
        partitions=["train", "embargo_1", "validation"],
    )
    a = analyze_relationship_direction(df, **kwargs)
    b = analyze_relationship_direction(df_with_target, **kwargs)
    c = analyze_relationship_direction(df_shuffled_target, **kwargs)
    assert _same_result(a, b)
    assert _same_result(a, c)


def test_no_target_dependency_signature():
    for fn in (
        build_relationship_frame,
        analyze_relationship_direction,
        relationship_row_coverage,
        overlap_diagnostic,
        select_high_fanout_anchors,
        compute_partner_prevalence,
        frequency_adjusted_overlap_diagnostic,
        evaluate_relationship_sufficiency,
        recommend_window_size,
        recommend_relationships,
    ):
        params = inspect.signature(fn).parameters
        assert "isFraud" not in params
        assert "target" not in params


# ---------------------------------------------------------------------------
# overlap_diagnostic (M5, descriptive only)
# ---------------------------------------------------------------------------


def _overlap_synthetic(c2_other_values):
    rows = [
        ("C1", 10, "X"), ("C1", 20, "Y"), ("C1", 30, "Z"), ("C1", 1000, "W"),
        ("C2", 15, c2_other_values[0]), ("C2", 25, c2_other_values[1]), ("C2", 35, c2_other_values[2]), ("C2", 1000, "T2"),
        ("C3", 5, "A0"),  # low-fanout anchor, single row -- excludes it from the top percentile
    ]
    return pd.DataFrame(rows, columns=["anchor", "TransactionDT", "other"])


def test_overlap_diagnostic_detects_overlap_vs_disjoint():
    # C1's causal partner set (dt<1000) is {X, Y, Z}. C2 shares "Z" -> overlap.
    df_overlap = _overlap_synthetic(["Z", "Q", "R"])
    res_overlap = overlap_diagnostic(df_overlap, anchor_col="anchor", other_col="other", dt_col="TransactionDT", high_fanout_percentile=100.0)
    assert res_overlap["n_anchors_total"] == 3
    assert res_overlap["n_high_fanout_anchors"] == 2  # C1, C2 both at max fanout (3); C3 excluded
    assert res_overlap["overlap_fraction_mean_pct"] == pytest.approx(100.0 / 3, rel=1e-6)
    assert res_overlap["clears_multi_hop_signal"] is True

    # C2's partners are fully disjoint from C1's -> zero overlap.
    df_disjoint = _overlap_synthetic(["M", "N", "O"])
    res_disjoint = overlap_diagnostic(df_disjoint, anchor_col="anchor", other_col="other", dt_col="TransactionDT", high_fanout_percentile=100.0)
    assert res_disjoint["n_high_fanout_anchors"] == 2
    assert res_disjoint["overlap_fraction_mean_pct"] == pytest.approx(0.0)
    assert res_disjoint["clears_multi_hop_signal"] is False


def test_overlap_diagnostic_stable_under_row_shuffle():
    df = _overlap_synthetic(["Z", "Q", "R"])
    shuffled = df.sample(frac=1.0, random_state=5).reset_index(drop=True)
    res_orig = overlap_diagnostic(df, "anchor", "other", "TransactionDT", high_fanout_percentile=100.0)
    res_shuf = overlap_diagnostic(shuffled, "anchor", "other", "TransactionDT", high_fanout_percentile=100.0)
    assert res_orig["overlap_fraction_mean_pct"] == pytest.approx(res_shuf["overlap_fraction_mean_pct"])
    assert res_orig["n_high_fanout_anchors"] == res_shuf["n_high_fanout_anchors"]


def test_overlap_diagnostic_excludes_own_final_bucket_from_partner_set():
    # C1's LAST row (dt=1000, other="W") must never appear in its own
    # partner set -- if it leaked in, C1's partner set would be {X,Y,Z,W}
    # (size 4) instead of {X,Y,Z} (size 3).
    df = _overlap_synthetic(["Z", "Q", "R"])
    res = overlap_diagnostic(df, "anchor", "other", "TransactionDT", high_fanout_percentile=100.0)
    # C1's overlap fraction is 1/3 (one shared partner "Z" out of 3) -- if
    # "W" had leaked into the denominator it would be 1/4 instead, changing
    # the mean away from exactly 100/3.
    assert res["overlap_fraction_mean_pct"] == pytest.approx(100.0 / 3, rel=1e-6)


def test_overlap_diagnostic_empty_frame():
    df = pd.DataFrame({"anchor": pd.Series(dtype="object"), "other": pd.Series(dtype="object"), "TransactionDT": pd.Series(dtype="int64")})
    res = overlap_diagnostic(df, "anchor", "other", "TransactionDT")
    assert res["n_anchors_total"] == 0
    assert res["clears_multi_hop_signal"] is False


def test_overlap_diagnostic_requires_columns():
    df = pd.DataFrame({"anchor": ["A"], "other": ["X"], "TransactionDT": [1]})
    with pytest.raises(ValueError):
        overlap_diagnostic(df, "no_such_col", "other", "TransactionDT")


# ---------------------------------------------------------------------------
# select_high_fanout_anchors (M5b anchor-selection fix)
# ---------------------------------------------------------------------------


def test_select_high_fanout_anchors_large_population_uses_percentile_derived_count():
    # 200 distinct-valued anchors: ceil(0.01*200)=2, but the floor (10) wins.
    fanout = pd.Series(range(1, 201))
    selected = select_high_fanout_anchors(fanout, percentile=99.0, min_anchor_count=10)
    assert len(selected) == 10
    assert set(selected) == set(range(190, 200))  # the 10 largest-index (largest-value) anchors


def test_select_high_fanout_anchors_small_population_floor():
    # 59 distinct-valued anchors mirrors P_emaildomain's real cardinality --
    # percentile-VALUE selection (old M5) collapses to 1 anchor here; the
    # rank-count floor must still select HIGH_FANOUT_MIN_ANCHOR_COUNT.
    fanout = pd.Series(list(range(1, 60)))
    selected = select_high_fanout_anchors(fanout, percentile=99.0, min_anchor_count=10)
    assert len(selected) == 10


def test_select_high_fanout_anchors_includes_ties_at_boundary():
    # 12 anchors, values [1]*8 + [5]*4 -- target count is 10 (floor), but the
    # cutoff value (1) is shared by all 8 low anchors, so every anchor >= 1
    # must be included (12 total), never split to hit exactly 10.
    fanout = pd.Series([1] * 8 + [5] * 4)
    selected = select_high_fanout_anchors(fanout, percentile=99.0, min_anchor_count=10)
    assert len(selected) == 12


def test_select_high_fanout_anchors_clamps_to_nonzero_fanout_anchors():
    # Only 3 of 20 anchors have any fan-out at all -- selection must not pad
    # past those 3 with zero-information anchors even though the floor is 10.
    fanout = pd.Series([0] * 17 + [1, 2, 3])
    selected = select_high_fanout_anchors(fanout, percentile=99.0, min_anchor_count=10)
    assert len(selected) == 3
    assert all(fanout.loc[a] > 0 for a in selected)


def test_select_high_fanout_anchors_empty_and_all_zero():
    assert select_high_fanout_anchors(pd.Series(dtype="int64")) == []
    assert select_high_fanout_anchors(pd.Series([0, 0, 0, 0, 0])) == []


# ---------------------------------------------------------------------------
# compute_partner_prevalence
# ---------------------------------------------------------------------------


def _prevalence_synthetic():
    # A's prior (dt<1000) partner set: {X}. B's: {X, Y}. C: single row, no
    # prior partner at all. D: {Y}. Final-bucket-only values (Z, W, V) must
    # never count toward prevalence.
    return pd.DataFrame(
        [
            ("A", 10, "X"), ("A", 1000, "Z"),
            ("B", 10, "X"), ("B", 20, "Y"), ("B", 1000, "W"),
            ("C", 5, "Q"),
            ("D", 10, "Y"), ("D", 1000, "V"),
        ],
        columns=["anchor", "TransactionDT", "other"],
    )


def test_compute_partner_prevalence_basic():
    df = _prevalence_synthetic()
    prevalence, n_anchors_total = compute_partner_prevalence(df, "anchor", "other", "TransactionDT")
    assert n_anchors_total == 4
    assert prevalence.loc["X"] == pytest.approx(0.5)  # A, B -> 2/4
    assert prevalence.loc["Y"] == pytest.approx(0.5)  # B, D -> 2/4
    assert "Z" not in prevalence.index  # A's own final-bucket value, never a prior partner
    assert "Q" not in prevalence.index  # C has no strictly-prior row at all


def test_compute_partner_prevalence_requires_columns():
    df = pd.DataFrame({"anchor": ["A"], "other": ["X"], "TransactionDT": [1]})
    with pytest.raises(ValueError):
        compute_partner_prevalence(df, "no_such_col", "other", "TransactionDT")


def test_compute_partner_prevalence_categorical_anchor_column():
    # Regression test: P_emaildomain (this project's real email_purchaser_node
    # anchor in the email_purchaser_to_* directions) is loaded as a pandas
    # categorical dtype. Series.map() on a categorical-dtype anchor column
    # previously preserved categorical structure for the numeric _anchor_last_dt
    # values it produced, which then raised
    # "Unordered Categoricals can only compare equality or not" on the `<`
    # comparison -- caught only when running the real E.1 pipeline.
    df = _prevalence_synthetic()
    df["anchor"] = df["anchor"].astype("category")
    prevalence, n_anchors_total = compute_partner_prevalence(df, "anchor", "other", "TransactionDT")
    assert n_anchors_total == 4
    assert prevalence.loc["X"] == pytest.approx(0.5)
    assert prevalence.loc["Y"] == pytest.approx(0.5)


def test_compute_partner_prevalence_row_order_independence():
    df = _prevalence_synthetic()
    shuffled = df.sample(frac=1.0, random_state=9).reset_index(drop=True)
    p1, n1 = compute_partner_prevalence(df, "anchor", "other", "TransactionDT")
    p2, n2 = compute_partner_prevalence(shuffled, "anchor", "other", "TransactionDT")
    assert n1 == n2
    pd.testing.assert_series_equal(p1.sort_index(), p2.sort_index())


# ---------------------------------------------------------------------------
# frequency_adjusted_overlap_diagnostic (M5b, decision-driving)
# ---------------------------------------------------------------------------


def _generic_hub_synthetic(n_common=40, n_singleton=10):
    # 40 "common" anchors all share the SAME 3 population-ubiquitous partner
    # values (present in 40/50 = 80% of all anchors) -- raw overlap among
    # them is 100%, exactly the pattern the real M5 review found for
    # browser/OS strings and email domains. 10 singleton anchors have no
    # prior partner at all (fan-out 0), giving a real population denominator.
    rows = []
    for i in range(n_common):
        a = f"C{i}"
        rows.append((a, 10, "G1"))
        rows.append((a, 20, "G2"))
        rows.append((a, 30, "G3"))
        rows.append((a, 1000, f"FINAL{i}"))  # excluded: own last dt
    for i in range(n_singleton):
        rows.append((f"S{i}", 5, f"SOLO{i}"))
    return pd.DataFrame(rows, columns=["anchor", "TransactionDT", "other"])


def test_frequency_adjusted_overlap_diagnostic_generic_hub_does_not_trigger_lift_signal():
    # False-positive guard: raw overlap is 100% (would trivially clear old
    # M5's 20% bar), but since the shared values are ubiquitous across the
    # population, the null model predicts ~100% too -- lift_ratio must stay
    # near 1 and NOT clear LIFT_SIGNAL_THRESHOLD.
    df = _generic_hub_synthetic()
    result = frequency_adjusted_overlap_diagnostic(df, "anchor", "other", "TransactionDT")
    assert result["n_anchors_total"] == 50
    assert result["n_high_fanout_anchors"] == 40  # all 40 common anchors tied at fan-out 3
    assert result["mean_observed_overlap_pct"] == pytest.approx(100.0)
    assert result["lift_ratio"] < LIFT_SIGNAL_THRESHOLD
    assert result["clears_lift_signal"] is False


def _rare_ring_synthetic():
    # 5 "ring" anchors share TWO values (RARE1, RARE2) found NOWHERE else in
    # a 500-anchor population (population prevalence 5/500 = 1% each) -- a
    # genuinely concentrated cross-anchor sharing pattern. 5 "filler"
    # anchors each have one unique, never-shared value (fan-out 1) so the
    # rank-count selection floor (10) is reached without diluting the ring
    # anchors' signal with more population noise than necessary. 490
    # background anchors have no prior partner at all.
    rows = []
    for i in range(5):
        a = f"RING{i}"
        rows.append((a, 10, "RARE1"))
        rows.append((a, 20, "RARE2"))
        rows.append((a, 1000, f"RING_FINAL{i}"))
    for i in range(5):
        a = f"FILL{i}"
        rows.append((a, 10, f"UNIQUE{i}"))
        rows.append((a, 1000, f"FILL_FINAL{i}"))
    for i in range(490):
        rows.append((f"BG{i}", 5, f"BGVAL{i}"))
    return pd.DataFrame(rows, columns=["anchor", "TransactionDT", "other"])


def test_frequency_adjusted_overlap_diagnostic_rare_shared_partner_triggers_lift_signal():
    # True positive: genuinely rare, concentrated cross-anchor sharing (1%
    # population prevalence, but 100% observed overlap among the 5 ring
    # anchors) must clear LIFT_SIGNAL_THRESHOLD.
    df = _rare_ring_synthetic()
    result = frequency_adjusted_overlap_diagnostic(df, "anchor", "other", "TransactionDT")
    assert result["n_anchors_total"] == 500
    assert result["n_high_fanout_anchors"] == 10  # 5 ring + 5 filler, no boundary overshoot
    assert result["lift_ratio"] >= LIFT_SIGNAL_THRESHOLD
    assert result["clears_lift_signal"] is True


def test_frequency_adjusted_overlap_diagnostic_small_population_selection_floor():
    # 59 anchors (mirrors P_emaildomain's real cardinality), strictly
    # decreasing distinct fan-out, all partner values anchor-unique (no real
    # sharing anywhere). M5's percentile-VALUE selection would collapse to a
    # single anchor here (see the E.1 critical-review conversation); M5b
    # must still select at least HIGH_FANOUT_MIN_ANCHOR_COUNT anchors and
    # produce a real (non-NaN) statistic.
    rows = []
    for i in range(59):
        a = f"E{i}"
        fanout = 59 - i
        for j in range(fanout):
            rows.append((a, 10 + j, f"P{i}_{j}"))
        rows.append((a, 1000, f"FINAL{i}"))
    df = pd.DataFrame(rows, columns=["anchor", "TransactionDT", "other"])
    result = frequency_adjusted_overlap_diagnostic(df, "anchor", "other", "TransactionDT")
    assert result["n_anchors_total"] == 59
    assert result["n_high_fanout_anchors"] >= HIGH_FANOUT_MIN_ANCHOR_COUNT
    assert not pd.isna(result["lift_ratio"])


def test_frequency_adjusted_overlap_diagnostic_row_order_independence():
    df = _rare_ring_synthetic()
    shuffled = df.sample(frac=1.0, random_state=21).reset_index(drop=True)
    res_orig = frequency_adjusted_overlap_diagnostic(df, "anchor", "other", "TransactionDT")
    res_shuf = frequency_adjusted_overlap_diagnostic(shuffled, "anchor", "other", "TransactionDT")
    assert res_orig["lift_ratio"] == pytest.approx(res_shuf["lift_ratio"])
    assert res_orig["n_high_fanout_anchors"] == res_shuf["n_high_fanout_anchors"]


def test_frequency_adjusted_overlap_diagnostic_requires_columns():
    df = pd.DataFrame({"anchor": ["A"], "other": ["X"], "TransactionDT": [1]})
    with pytest.raises(ValueError):
        frequency_adjusted_overlap_diagnostic(df, "no_such_col", "other", "TransactionDT")


def test_frequency_adjusted_overlap_diagnostic_empty_frame():
    df = pd.DataFrame(
        {"anchor": pd.Series(dtype="object"), "other": pd.Series(dtype="object"), "TransactionDT": pd.Series(dtype="int64")}
    )
    result = frequency_adjusted_overlap_diagnostic(df, "anchor", "other", "TransactionDT")
    assert result["n_anchors_total"] == 0
    assert result["clears_lift_signal"] is False


def test_frequency_adjusted_overlap_diagnostic_no_target_dependency_behavioral():
    df = _rare_ring_synthetic()
    df_with_target = df.copy()
    df_with_target["isFraud"] = 0
    a = frequency_adjusted_overlap_diagnostic(df, "anchor", "other", "TransactionDT")
    b = frequency_adjusted_overlap_diagnostic(df_with_target, "anchor", "other", "TransactionDT")
    assert a["lift_ratio"] == pytest.approx(b["lift_ratio"])
    assert a["n_high_fanout_anchors"] == b["n_high_fanout_anchors"]


# ---------------------------------------------------------------------------
# evaluate_relationship_sufficiency
# ---------------------------------------------------------------------------


def _good_direction_result(overall_pct=80.0, partition_pcts=None, dominant_pcts=None):
    partition_pcts = partition_pcts or {"train": 80.0, "embargo_1": 78.0, "validation": 75.0}
    dominant_pcts = dominant_pcts if dominant_pcts is not None else [78.0, 76.0, 70.0, 65.0]
    return {
        "unbounded": {
            "sufficiency_overall": {2: overall_pct},
            "sufficiency_by_partition": [{"partition": p, "pct_sufficient_ge_2": v} for p, v in partition_pcts.items()],
            "dominant_anchor_exclusion_sensitivity": [
                {"top_k_excluded": k, "pct_sufficient_ge_2": v} for k, v in zip([1, 3, 5, 10], dominant_pcts)
            ],
        }
    }


def _coverage(pct_rows_valid=90.0):
    return {"pct_rows_valid": pct_rows_valid}


def test_evaluate_relationship_sufficiency_passes_when_stable_and_robust():
    ev = evaluate_relationship_sufficiency(_good_direction_result(), _coverage(), decision_threshold=2)
    assert ev["fanout_density_ok"]
    assert ev["partition_stability_ok"]
    assert ev["dominant_anchor_robustness_ok"]
    assert ev["row_coverage_ok"]
    assert ev["is_suitable"]


def test_evaluate_relationship_sufficiency_fails_on_low_row_coverage():
    ev = evaluate_relationship_sufficiency(_good_direction_result(), _coverage(pct_rows_valid=5.0), decision_threshold=2)
    assert not ev["row_coverage_ok"]
    assert not ev["is_suitable"]


def test_evaluate_relationship_sufficiency_fails_on_low_density():
    result = _good_direction_result(overall_pct=1.0, partition_pcts={"train": 1.0, "embargo_1": 0.9, "validation": 0.8})
    ev = evaluate_relationship_sufficiency(result, _coverage(), decision_threshold=2)
    assert not ev["fanout_density_ok"]
    assert not ev["is_suitable"]


def test_evaluate_relationship_sufficiency_fails_on_dominant_collapse():
    result = _good_direction_result(dominant_pcts=[5.0, 3.0, 1.0, 0.5])
    ev = evaluate_relationship_sufficiency(result, _coverage(), decision_threshold=2)
    assert not ev["dominant_anchor_robustness_ok"]
    assert not ev["is_suitable"]


def test_evaluate_relationship_sufficiency_fails_on_partition_concentration():
    result = _good_direction_result(partition_pcts={"train": 80.0, "embargo_1": 5.0, "validation": 3.0})
    ev = evaluate_relationship_sufficiency(result, _coverage(), decision_threshold=2)
    assert not ev["partition_stability_ok"]
    assert not ev["is_suitable"]


# ---------------------------------------------------------------------------
# recommend_window_size
# ---------------------------------------------------------------------------


def _direction_with_windows(unbounded_pct, windowed_pcts):
    return {
        "unbounded": {"sufficiency_overall": {2: unbounded_pct}},
        "windowed": {w: {"sufficiency_overall": {2: pct}} for w, pct in windowed_pcts.items()},
    }


def test_recommend_window_size_picks_largest_stable_window():
    result = _direction_with_windows(80.0, {5: 79.0, 10: 60.0, 20: 45.0, 50: 30.0})
    w = recommend_window_size(result, decision_threshold=2, window_candidates=[5, 10, 20, 50], min_partition_relative_pct=50.0)
    assert w == 20  # 30 < 80*0.5=40 fails at w=50; 45 >= 40 passes at w=20


def test_recommend_window_size_returns_none_when_unbounded_is_zero():
    result = _direction_with_windows(0.0, {5: 0.0})
    assert recommend_window_size(result, decision_threshold=2, window_candidates=[5]) is None


def test_recommend_window_size_returns_none_when_no_window_clears_bar():
    result = _direction_with_windows(80.0, {5: 10.0, 10: 5.0})
    assert recommend_window_size(result, decision_threshold=2, window_candidates=[5, 10]) is None


# ---------------------------------------------------------------------------
# recommend_relationships
# ---------------------------------------------------------------------------


def test_recommend_relationships_true_when_signal_present():
    # M5's raw overlap (clears_multi_hop_signal) is deliberately set opposite
    # to M5b's lift signal on both directions below, to prove the aggregate
    # is driven by M5b (frequency_adjusted_overlap/clears_lift_signal), never by M5.
    all_dirs = {
        "payment_to_device": {
            "evaluation": {"is_suitable": True},
            "overlap": {"clears_multi_hop_signal": False, "overlap_fraction_mean_pct": 10.0},
            "frequency_adjusted_overlap": {"clears_lift_signal": True, "lift_ratio": 3.0},
            "recommended_window": 20,
        },
        "device_to_payment": {
            "evaluation": {"is_suitable": True},
            "overlap": {"clears_multi_hop_signal": True, "overlap_fraction_mean_pct": 95.0},
            "frequency_adjusted_overlap": {"clears_lift_signal": False, "lift_ratio": 1.1},
            "recommended_window": None,
        },
    }
    rec = recommend_relationships(all_dirs)
    assert rec["recommend_union_find_for_e2"] is True
    assert rec["per_direction"]["payment_to_device"]["is_suitable"] is True
    assert rec["per_direction"]["payment_to_device"]["recommended_window"] == 20
    assert rec["per_direction"]["payment_to_device"]["lift_ratio"] == 3.0


def test_recommend_relationships_false_when_no_signal_even_if_raw_overlap_high():
    # High M5 raw overlap alone (95%) must NOT trigger the recommendation
    # when M5b shows no excess beyond population expectation.
    all_dirs = {
        "payment_to_device": {
            "evaluation": {"is_suitable": True},
            "overlap": {"clears_multi_hop_signal": True, "overlap_fraction_mean_pct": 95.0},
            "frequency_adjusted_overlap": {"clears_lift_signal": False, "lift_ratio": 1.0},
            "recommended_window": None,
        },
    }
    rec = recommend_relationships(all_dirs)
    assert rec["recommend_union_find_for_e2"] is False


def test_recommend_relationships_ignores_signal_from_unsuitable_direction():
    # A direction that ISN'T suitable but happens to clear the M5b lift
    # signal should not trigger the aggregate recommendation.
    all_dirs = {
        "payment_to_device": {
            "evaluation": {"is_suitable": False},
            "overlap": {"clears_multi_hop_signal": True, "overlap_fraction_mean_pct": 90.0},
            "frequency_adjusted_overlap": {"clears_lift_signal": True, "lift_ratio": 5.0},
            "recommended_window": None,
        },
    }
    rec = recommend_relationships(all_dirs)
    assert rec["recommend_union_find_for_e2"] is False

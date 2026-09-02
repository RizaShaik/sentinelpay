import math

import numpy as np
import pandas as pd
import pytest

from sentinelpay.target_history import (
    OUTPUT_COLUMNS,
    SUFFICIENT_HISTORY_THRESHOLD,
    TRAIN_RECIPIENT_PARTITIONS,
    VALIDATION_SOURCE_PARTITIONS,
    build_eligible_pools,
    compute_prior_fraud_rate,
)


def _hand_scenario():
    # Deliberately synthetic group column (never a real proxy key name at
    # this module-test level, matching sentinelpay.data.history's own
    # convention) -- all rows in one partition ("train") so the
    # allowed_source_partitions contract is trivially satisfied; the
    # partition-contract itself is tested separately below.
    return pd.DataFrame(
        {
            "grp": ["A", "B", "A", "A"],
            "TransactionDT": [100, 150, 200, 300],
            "isFraud": [1, 0, 0, 1],
            "partition": ["train"] * 4,
        }
    )


def test_hand_computed_smoothing_and_cold_start():
    df = _hand_scenario()
    out = compute_prior_fraud_rate(
        df, allowed_source_partitions={"train"}, group_col="grp", dt_col="TransactionDT", k=2.0
    )

    # r0 (A, dt=100): first row ever seen anywhere in the pool -> global
    # pool empty -> global_cold_start, smoothed is NaN.
    assert out.loc[0, "payment_proxy_prior_fraud_count"] == 0
    assert out.loc[0, "payment_proxy_prior_event_count"] == 0
    assert out.loc[0, "global_cold_start"] == True  # noqa: E712
    assert pd.isna(out.loc[0, "global_prior_fraud_rate"])
    assert pd.isna(out.loc[0, "payment_proxy_prior_fraud_rate_raw"])
    assert pd.isna(out.loc[0, "payment_proxy_prior_fraud_rate_smoothed"])

    # r1 (B, dt=150): own key B has zero prior history, but the GLOBAL pool
    # now has 1 prior event (r0, isFraud=1) -> global_rate=1.0. Cold-start
    # formula: smoothed must equal global_prior_fraud_rate exactly.
    assert out.loc[1, "payment_proxy_prior_event_count"] == 0
    assert out.loc[1, "global_cold_start"] == False  # noqa: E712
    assert out.loc[1, "global_prior_fraud_rate"] == pytest.approx(1.0)
    assert out.loc[1, "payment_proxy_prior_fraud_rate_smoothed"] == pytest.approx(1.0)
    assert pd.isna(out.loc[1, "payment_proxy_prior_fraud_rate_raw"])

    # r2 (A, dt=200): own-key prior = {r0} -> fraud_count=1, event_count=1,
    # raw=1.0. Global prior = {r0, r1} -> fraud=1, count=2 -> rate=0.5.
    # smoothed (k=2) = (1 + 2*0.5) / (1 + 2) = 2/3.
    assert out.loc[2, "payment_proxy_prior_fraud_count"] == 1
    assert out.loc[2, "payment_proxy_prior_event_count"] == 1
    assert out.loc[2, "payment_proxy_prior_fraud_rate_raw"] == pytest.approx(1.0)
    assert out.loc[2, "global_prior_fraud_rate"] == pytest.approx(0.5)
    assert out.loc[2, "payment_proxy_prior_fraud_rate_smoothed"] == pytest.approx(2.0 / 3.0)

    # r3 (A, dt=300): own-key prior = {r0, r2} -> fraud=1, count=2,
    # raw=0.5. Global prior = {r0, r1, r2} -> fraud=1, count=3 -> rate=1/3.
    # smoothed (k=2) = (1 + 2*(1/3)) / (2 + 2) = (1 + 2/3) / 4.
    assert out.loc[3, "payment_proxy_prior_fraud_count"] == 1
    assert out.loc[3, "payment_proxy_prior_event_count"] == 2
    assert out.loc[3, "payment_proxy_prior_fraud_rate_raw"] == pytest.approx(0.5)
    assert out.loc[3, "global_prior_fraud_rate"] == pytest.approx(1.0 / 3.0)
    assert out.loc[3, "payment_proxy_prior_fraud_rate_smoothed"] == pytest.approx((1.0 + 2.0 / 3.0) / 4.0)

    assert list(out.columns) == OUTPUT_COLUMNS


def test_sufficient_target_history_flag_uses_own_key_count_only():
    df = _hand_scenario()
    out = compute_prior_fraud_rate(df, allowed_source_partitions={"train"}, group_col="grp", dt_col="TransactionDT")
    # None of these rows reach SUFFICIENT_HISTORY_THRESHOLD (5) own-key prior events.
    assert (~out["sufficient_target_history"]).all()

    many = pd.DataFrame(
        {
            "grp": ["A"] * 7,
            "TransactionDT": list(range(7)),
            "isFraud": [0, 1, 0, 1, 0, 0, 1],
            "partition": ["train"] * 7,
        }
    )
    out2 = compute_prior_fraud_rate(many, allowed_source_partitions={"train"}, group_col="grp", dt_col="TransactionDT")
    # Row at position 5 (0-indexed) has 5 strictly-prior same-key events -> sufficient.
    assert out2.loc[5, "payment_proxy_prior_event_count"] == 5
    assert out2.loc[5, "sufficient_target_history"] == True  # noqa: E712
    assert SUFFICIENT_HISTORY_THRESHOLD == 5


def test_requires_columns():
    df = _hand_scenario()
    for col in ["grp", "TransactionDT", "isFraud", "partition"]:
        bad = df.drop(columns=[col])
        with pytest.raises(ValueError):
            compute_prior_fraud_rate(bad, allowed_source_partitions={"train"}, group_col="grp", dt_col="TransactionDT")


# ---------------------------------------------------------------------------
# Explicit source-partition contract (no incidental filtering)
# ---------------------------------------------------------------------------


def test_allowed_source_partitions_contract_rejects_disallowed_partition():
    df = pd.DataFrame(
        {
            "grp": ["A", "A"],
            "TransactionDT": [100, 200],
            "isFraud": [1, 0],
            "partition": ["train", "embargo_1"],
        }
    )
    with pytest.raises(ValueError):
        compute_prior_fraud_rate(df, allowed_source_partitions={"train"}, group_col="grp", dt_col="TransactionDT")
    with pytest.raises(ValueError):
        compute_prior_fraud_rate(
            df, allowed_source_partitions={"train", "validation"}, group_col="grp", dt_col="TransactionDT"
        )


def test_allowed_source_partitions_contract_accepts_exact_match():
    df = pd.DataFrame(
        {
            "grp": ["A", "A"],
            "TransactionDT": [100, 200],
            "isFraud": [1, 0],
            "partition": ["train", "validation"],
        }
    )
    out = compute_prior_fraud_rate(
        df, allowed_source_partitions={"train", "validation"}, group_col="grp", dt_col="TransactionDT"
    )
    assert len(out) == 2


# ---------------------------------------------------------------------------
# build_eligible_pools -- holdout sealing + embargo exclusion by construction
# ---------------------------------------------------------------------------


def _all_five_partitions_frame():
    return pd.DataFrame(
        {
            "grp": ["K"] * 5,
            "TransactionDT": [10, 50, 60, 70, 80],
            "isFraud": [1, 1, 1, 0, 0],
            "partition": ["train", "embargo_1", "validation", "embargo_2", "holdout"],
        }
    )


def test_build_eligible_pools_excludes_embargo_and_holdout_by_construction():
    df = _all_five_partitions_frame()
    train_pool, validation_pool = build_eligible_pools(df)

    assert set(train_pool["partition"].unique()) == TRAIN_RECIPIENT_PARTITIONS
    assert set(validation_pool["partition"].unique()) <= VALIDATION_SOURCE_PARTITIONS
    assert "embargo_1" not in validation_pool["partition"].unique()
    assert "embargo_2" not in validation_pool["partition"].unique()
    assert "holdout" not in validation_pool["partition"].unique()
    assert "holdout" not in train_pool["partition"].unique()
    assert len(train_pool) == 1
    assert len(validation_pool) == 2  # train + validation rows only


def test_build_eligible_pools_requires_partition_column():
    df = _all_five_partitions_frame().drop(columns=["partition"])
    with pytest.raises(ValueError):
        build_eligible_pools(df)


# ---------------------------------------------------------------------------
# The specific required adversarial test: an embargo_1 fraud label must not
# affect a later validation row sharing the same key, while an earlier
# validation label must.
# ---------------------------------------------------------------------------


def test_embargo_1_label_excluded_but_earlier_validation_label_included():
    df = pd.DataFrame(
        {
            "grp": ["K", "K", "K", "K"],
            "TransactionDT": [10, 50, 60, 70],
            "isFraud": [1, 1, 1, 0],
            "partition": ["train", "embargo_1", "validation", "validation"],
        }
        # t1 (train, dt=10, fraud=1)      -- MUST be visible to v2
        # e1 (embargo_1, dt=50, fraud=1)  -- MUST NOT be visible to v2
        # v1 (validation, dt=60, fraud=1) -- MUST be visible to v2 (earlier validation)
        # v2 (validation, dt=70)          -- row under test
    )
    _, validation_pool = build_eligible_pools(df)
    # embargo_1 physically absent from the pool passed to the causal primitive.
    assert "embargo_1" not in validation_pool["partition"].unique()

    out = compute_prior_fraud_rate(
        validation_pool,
        allowed_source_partitions=VALIDATION_SOURCE_PARTITIONS,
        group_col="grp",
        dt_col="TransactionDT",
    )
    v2_idx = validation_pool.index[validation_pool["TransactionDT"] == 70][0]
    # If embargo_1's fraud label had leaked in, this would be 3, not 2.
    assert out.loc[v2_idx, "payment_proxy_prior_fraud_count"] == 2
    assert out.loc[v2_idx, "payment_proxy_prior_event_count"] == 2


# ---------------------------------------------------------------------------
# Same-TransactionDT ties never see each other (re-asserted at this module's
# own integration level, not only trusted transitively from history.py)
# ---------------------------------------------------------------------------


def test_same_time_ties_never_see_each_other():
    df = pd.DataFrame(
        {
            "grp": ["A", "A"],
            "TransactionDT": [500, 500],
            "isFraud": [1, 0],
            "partition": ["train", "train"],
        }
    )
    out = compute_prior_fraud_rate(df, allowed_source_partitions={"train"}, group_col="grp", dt_col="TransactionDT")
    assert (out["payment_proxy_prior_event_count"] == 0).all()
    assert (out["payment_proxy_prior_fraud_count"] == 0).all()
    assert (out["global_cold_start"] == True).all()  # noqa: E712


# ---------------------------------------------------------------------------
# Row-order independence
# ---------------------------------------------------------------------------


def test_row_order_independence():
    df = _hand_scenario()
    shuffled = df.sample(frac=1.0, random_state=9).reset_index(drop=True)

    out_orig = compute_prior_fraud_rate(df, allowed_source_partitions={"train"}, group_col="grp", dt_col="TransactionDT")
    out_shuf = compute_prior_fraud_rate(
        shuffled, allowed_source_partitions={"train"}, group_col="grp", dt_col="TransactionDT"
    )

    def _key(frame, i):
        return (frame.loc[i, "grp"], frame.loc[i, "TransactionDT"])

    lookup = {_key(df, i): tuple(out_orig.loc[i].fillna("NA")) for i in df.index}
    for i in shuffled.index:
        assert lookup[_key(shuffled, i)] == tuple(out_shuf.loc[i].fillna("NA"))


# ---------------------------------------------------------------------------
# Future perturbation invariance
# ---------------------------------------------------------------------------


def test_future_perturbation_does_not_change_earlier_rows():
    df = _hand_scenario()
    before = compute_prior_fraud_rate(df, allowed_source_partitions={"train"}, group_col="grp", dt_col="TransactionDT")

    df_mutated = df.copy()
    last_idx = df_mutated["TransactionDT"].idxmax()
    df_mutated.loc[last_idx, "isFraud"] = 1 - df_mutated.loc[last_idx, "isFraud"]
    df_mutated.loc[last_idx, "TransactionDT"] = 999_999
    after = compute_prior_fraud_rate(
        df_mutated, allowed_source_partitions={"train"}, group_col="grp", dt_col="TransactionDT"
    )

    for idx in [i for i in df.index if i != last_idx]:
        for col in OUTPUT_COLUMNS:
            b, a = before.loc[idx, col], after.loc[idx, col]
            assert (isinstance(b, float) and math.isnan(b) and isinstance(a, float) and math.isnan(a)) or b == a

    new_row = pd.DataFrame({"grp": ["A"], "TransactionDT": [10_000], "isFraud": [1], "partition": ["train"]})
    df_extended = pd.concat([df, new_row], ignore_index=True)
    after_append = compute_prior_fraud_rate(
        df_extended, allowed_source_partitions={"train"}, group_col="grp", dt_col="TransactionDT"
    )
    for idx in df.index:
        for col in OUTPUT_COLUMNS:
            b, a = before.loc[idx, col], after_append.loc[idx, col]
            assert (isinstance(b, float) and math.isnan(b) and isinstance(a, float) and math.isnan(a)) or b == a


# ---------------------------------------------------------------------------
# Explicit target-dependency contract (the inverse of every other module's
# no-target-dependency test -- this module DOES and MUST depend on the
# target column's actual values)
# ---------------------------------------------------------------------------


def test_explicit_target_dependency_contract():
    df = _hand_scenario()
    df_alt_target = df.copy()
    df_alt_target["isFraud"] = 1 - df_alt_target["isFraud"]  # flip every label

    a = compute_prior_fraud_rate(df, allowed_source_partitions={"train"}, group_col="grp", dt_col="TransactionDT")
    b = compute_prior_fraud_rate(
        df_alt_target, allowed_source_partitions={"train"}, group_col="grp", dt_col="TransactionDT"
    )
    # Output MUST differ when the target values differ -- proves this
    # function genuinely reads and depends on target_col, unlike every
    # sentinelpay.data.history / sentinelpay.eda.*_analysis function.
    assert not a["payment_proxy_prior_fraud_rate_smoothed"].equals(b["payment_proxy_prior_fraud_rate_smoothed"])


# ---------------------------------------------------------------------------
# Brute-force oracle: from-scratch reimplementation operating only on rows
# physically present in pool_df (partition-agnostic, matching
# compute_prior_fraud_rate's own "no incidental filtering" contract).
# ---------------------------------------------------------------------------


def _brute_force_prior_fraud_rate(pool_df, group_col, dt_col, target_col, k, min_history_threshold):
    rows = []
    for i in pool_df.index:
        gi, dti = pool_df.loc[i, group_col], pool_df.loc[i, dt_col]
        prior_all = pool_df[pool_df[dt_col] < dti]
        prior_own = prior_all[prior_all[group_col] == gi]

        prior_fraud_count = int(prior_own[target_col].sum())
        prior_event_count = int(len(prior_own))
        global_prior_fraud_count = int(prior_all[target_col].sum())
        global_prior_event_count = int(len(prior_all))

        global_cold_start = global_prior_event_count == 0
        global_rate = float("nan") if global_cold_start else global_prior_fraud_count / global_prior_event_count
        raw = float("nan") if prior_event_count == 0 else prior_fraud_count / prior_event_count
        smoothed = (prior_fraud_count + k * global_rate) / (prior_event_count + k)
        sufficient = prior_event_count >= min_history_threshold

        rows.append(
            {
                "payment_proxy_prior_fraud_count": prior_fraud_count,
                "payment_proxy_prior_event_count": prior_event_count,
                "global_prior_fraud_rate": global_rate,
                "payment_proxy_prior_fraud_rate_raw": raw,
                "payment_proxy_prior_fraud_rate_smoothed": smoothed,
                "sufficient_target_history": sufficient,
                "global_cold_start": global_cold_start,
            }
        )
    return pd.DataFrame(rows, index=pool_df.index)[OUTPUT_COLUMNS]


def test_matches_brute_force_on_random_data_with_many_ties():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "grp": rng.choice(["A", "B", "C", "D"], size=n),
            "TransactionDT": rng.integers(0, 12, size=n),  # small range forces heavy ties
            "isFraud": rng.integers(0, 2, size=n),
            "partition": rng.choice(["train", "validation"], size=n),
        }
    )
    fast = compute_prior_fraud_rate(
        df, allowed_source_partitions={"train", "validation"}, group_col="grp", dt_col="TransactionDT", k=3.0
    )
    slow = _brute_force_prior_fraud_rate(df, "grp", "TransactionDT", "isFraud", k=3.0, min_history_threshold=5)
    pd.testing.assert_frame_equal(
        fast.astype({"sufficient_target_history": bool, "global_cold_start": bool}), slow, check_dtype=False
    )


def test_matches_brute_force_row_order_shuffled():
    rng = np.random.default_rng(1)
    n = 45
    df = pd.DataFrame(
        {
            "grp": rng.choice(["A", "B", "C"], size=n),
            "TransactionDT": rng.integers(0, 9, size=n),
            "isFraud": rng.integers(0, 2, size=n),
            "partition": rng.choice(["train", "validation"], size=n),
        }
    )
    shuffled = df.sample(frac=1.0, random_state=42)  # keeps original index
    fast = compute_prior_fraud_rate(
        shuffled, allowed_source_partitions={"train", "validation"}, group_col="grp", dt_col="TransactionDT", k=3.0
    )
    slow = _brute_force_prior_fraud_rate(shuffled, "grp", "TransactionDT", "isFraud", k=3.0, min_history_threshold=5)
    pd.testing.assert_frame_equal(
        fast.astype({"sufficient_target_history": bool, "global_cold_start": bool}), slow, check_dtype=False
    )

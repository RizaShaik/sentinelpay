import pandas as pd
import pytest

from sentinelpay.config import DetectionConfig, load_config
from sentinelpay.data.split import load_split_config
from sentinelpay.detection import compute_behavioral_change_score
from sentinelpay.inference.state import (
    PHASE_D_STATE_SOURCE_PARTITIONS,
    PHASE_F_STATE_SOURCE_PARTITIONS,
    InferenceState,
    build_initial_state,
    load_state,
    record_observed_transactions,
    save_state,
    update_resolved_labels,
)
from sentinelpay.model_features import PAYMENT_GROUP_COL


def _synthetic_five_partition_frame(config):
    seconds_per_day = config.seconds_per_day
    days = {"train": 5, "embargo_1": 133, "validation": 150, "embargo_2": 165, "holdout": 175}
    rows = []
    for partition, day in days.items():
        for i in range(3):
            rows.append(
                {
                    "TransactionID": len(rows) + 1,
                    "TransactionDT": day * seconds_per_day + i,
                    "TransactionAmt": 10.0 + i,
                    "isFraud": 1 if i == 0 else 0,  # 1 fraud per partition (3 rows each)
                    **{col: 1 for col in config.payment_proxy_key_columns},
                }
            )
    return pd.DataFrame(rows)


def test_build_initial_state_partition_scope(monkeypatch):
    config = load_config()
    split_config = load_split_config()
    synthetic = _synthetic_five_partition_frame(config)

    def fake_load_transaction_columns(split, columns, config=None):
        return synthetic[columns].copy()

    import sentinelpay.eda.run_phase_h as run_phase_h_module

    monkeypatch.setattr(run_phase_h_module, "load_transaction_columns", fake_load_transaction_columns)

    state = build_initial_state(config=config, split_config=split_config)

    # Phase D buffer: ALL FIVE partitions' rows present (15 rows, 3 per partition).
    assert len(state.phase_d_buffer) == 15

    # Phase F: only train+validation+holdout (9 rows: 3 partitions x 3 rows),
    # embargo_1/embargo_2 (6 rows, 2 fraud) excluded.
    assert state.phase_f_global_event_count == 9
    assert state.phase_f_global_fraud_count == 3  # 1 fraud per included partition x 3 partitions

    assert state.phase_f_counts.index.name == PAYMENT_GROUP_COL

    # Registries are seeded from the SAME pools -- all 15 TransactionIDs
    # (1..15, assigned in partition-row order by the synthetic fixture) for
    # Phase D; only the 9 belonging to train/validation/holdout for Phase F
    # (rows 1-3=train, 4-6=embargo_1, 7-9=validation, 10-12=embargo_2,
    # 13-15=holdout).
    assert set(state.phase_d_processed.index) == set(range(1, 16))
    assert set(state.phase_f_processed.index) == {1, 2, 3, 7, 8, 9, 13, 14, 15}


def test_build_initial_state_metadata_fields(monkeypatch):
    config = load_config()
    split_config = load_split_config()
    synthetic = _synthetic_five_partition_frame(config)

    def fake_load_transaction_columns(split, columns, config=None):
        return synthetic[columns].copy()

    import sentinelpay.eda.run_phase_h as run_phase_h_module

    monkeypatch.setattr(run_phase_h_module, "load_transaction_columns", fake_load_transaction_columns)

    state = build_initial_state(config=config, split_config=split_config)
    meta = state.metadata

    assert meta["model_fit_source"] == "train partition ONLY (Phase G/H regime) -- see sentinelpay.inference.artifacts"
    assert "train + validation + holdout ONLY" in meta["phase_f_state_source"]
    assert set(meta["phase_f_embargo_partitions_excluded"]) == {"embargo_1", "embargo_2"}
    assert set(meta["phase_d_partitions_included"]) == PHASE_D_STATE_SOURCE_PARTITIONS
    assert "resolution-only" in meta["phase_f_update_policy"]
    assert "NOT rerun" in meta["holdout_usage_note"]
    assert PHASE_F_STATE_SOURCE_PARTITIONS == {"train", "validation", "holdout"}
    assert meta["n_ids_phase_d_observed"] == 15
    assert meta["n_ids_phase_f_resolved"] == 9


def test_save_load_state_roundtrip(tmp_path):
    phase_d_buffer = pd.DataFrame({PAYMENT_GROUP_COL: ["K1", "K1"], "TransactionDT": [10, 20], "TransactionAmt": [1.0, 2.0]})
    phase_f_counts = pd.DataFrame({"fraud_count": [1], "event_count": [3]}, index=pd.Index(["K1"], name=PAYMENT_GROUP_COL))
    phase_d_processed = pd.DataFrame(
        {PAYMENT_GROUP_COL: ["K1", "K1"], "TransactionDT": [10, 20], "TransactionAmt": [1.0, 2.0]},
        index=pd.Index([10, 20], name="TransactionID"),
    )
    phase_f_processed = pd.DataFrame(
        {PAYMENT_GROUP_COL: ["K1"], "isFraud": [1]}, index=pd.Index([20], name="TransactionID")
    )
    state = InferenceState(
        phase_d_buffer=phase_d_buffer,
        phase_f_counts=phase_f_counts,
        phase_f_global_fraud_count=5,
        phase_f_global_event_count=20,
        phase_d_processed=phase_d_processed,
        phase_f_processed=phase_f_processed,
        metadata={"note": "test"},
    )
    save_state(state, tmp_path / "state")
    loaded = load_state(tmp_path / "state")

    pd.testing.assert_frame_equal(loaded.phase_d_buffer, state.phase_d_buffer)
    pd.testing.assert_frame_equal(loaded.phase_f_counts, state.phase_f_counts)
    assert loaded.phase_f_global_fraud_count == 5
    assert loaded.phase_f_global_event_count == 20
    pd.testing.assert_frame_equal(loaded.phase_d_processed, state.phase_d_processed)
    pd.testing.assert_frame_equal(loaded.phase_f_processed, state.phase_f_processed)
    assert loaded.metadata["note"] == "test"


def _base_state():
    phase_d_buffer = pd.DataFrame(columns=[PAYMENT_GROUP_COL, "TransactionDT", "TransactionAmt"])
    phase_f_counts = pd.DataFrame({"fraud_count": [1], "event_count": [3]}, index=pd.Index(["K1"], name=PAYMENT_GROUP_COL))
    return InferenceState(
        phase_d_buffer=phase_d_buffer,
        phase_f_counts=phase_f_counts,
        phase_f_global_fraud_count=5,
        phase_f_global_event_count=20,
        metadata={},
    )


def test_update_resolved_labels_increments_correctly_direct_key():
    state = _base_state()
    resolved = pd.DataFrame({PAYMENT_GROUP_COL: ["K1", "K1", "K2"], "isFraud": [1, 0, 1], "TransactionID": [101, 102, 103]})
    new_state = update_resolved_labels(state, resolved)

    assert new_state.phase_f_counts.loc["K1", "fraud_count"] == 2  # 1 (existing) + 1 (new)
    assert new_state.phase_f_counts.loc["K1", "event_count"] == 5  # 3 (existing) + 2 (new)
    assert new_state.phase_f_counts.loc["K2", "fraud_count"] == 1  # brand new key
    assert new_state.phase_f_counts.loc["K2", "event_count"] == 1
    assert new_state.phase_f_global_fraud_count == 5 + 2  # 2 fraud in this batch
    assert new_state.phase_f_global_event_count == 20 + 3
    assert set(new_state.phase_f_processed.index) == {101, 102, 103}


def test_update_resolved_labels_does_not_mutate_input_state():
    state = _base_state()
    original_counts = state.phase_f_counts.copy()
    resolved = pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "isFraud": [1], "TransactionID": [104]})
    update_resolved_labels(state, resolved)
    pd.testing.assert_frame_equal(state.phase_f_counts, original_counts)
    assert state.phase_f_global_fraud_count == 5
    assert state.phase_f_global_event_count == 20
    assert len(state.phase_f_processed) == 0


def test_update_resolved_labels_via_raw_key_components():
    config = load_config()
    state = _base_state()
    resolved = pd.DataFrame(
        {
            **{col: [1] for col in config.payment_proxy_key_columns},
            "isFraud": [1],
            "TransactionID": [105],
        }
    )
    new_state = update_resolved_labels(state, resolved, config=config)
    # The raw components all equal 1 -- same key construction build_group_key
    # would use ("1|1|1|1|1"); confirm the resulting key received the update.
    assert new_state.phase_f_global_event_count == 21
    assert new_state.phase_f_global_fraud_count == 6


def test_update_resolved_labels_does_not_touch_phase_d_buffer():
    state = _base_state()
    resolved = pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "isFraud": [1], "TransactionID": [106]})
    new_state = update_resolved_labels(state, resolved)
    pd.testing.assert_frame_equal(new_state.phase_d_buffer, state.phase_d_buffer)
    assert len(new_state.phase_d_processed) == len(state.phase_d_processed) == 0


def test_update_resolved_labels_requires_isfraud_column():
    state = _base_state()
    # TransactionID present -- isolates this check to "isFraud missing"
    # specifically, not conflated with the separate TransactionID check.
    resolved = pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionID": [999]})
    with pytest.raises(ValueError):
        update_resolved_labels(state, resolved)


def test_update_resolved_labels_requires_transaction_id():
    """TransactionID AVAILABILITY CONTRACT (explicit, dedicated proof): the
    state-MUTATING half of the contract -- unlike scoring (see
    tests/test_inference_scoring.py::test_score_transaction_succeeds_without_transaction_id),
    update_resolved_labels must fail CLEARLY, not silently proceed
    non-idempotently, when TransactionID is absent."""
    state = _base_state()
    resolved = pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "isFraud": [1]})  # no TransactionID
    with pytest.raises(ValueError):
        update_resolved_labels(state, resolved)


# ---------------------------------------------------------------------------
# record_observed_transactions -- Phase D occurrence-only state update.
# ---------------------------------------------------------------------------


def _detection_config():
    return DetectionConfig(
        min_history_for_score=1,
        window_size_events=10,
        modified_zscore_scale_constant=0.6745,
        modified_zscore_threshold=3.5,
        zero_mad_epsilon=1e-9,
    )


def _empty_d_buffer():
    return pd.DataFrame(columns=[PAYMENT_GROUP_COL, "TransactionDT", "TransactionAmt"])


def test_record_observed_transactions_grows_phase_d_buffer():
    state = _base_state()
    assert len(state.phase_d_buffer) == 0

    observed = pd.DataFrame(
        {PAYMENT_GROUP_COL: ["K1", "K2"], "TransactionDT": [100, 200], "TransactionAmt": [10.0, 20.0], "TransactionID": [201, 202]}
    )
    new_state = record_observed_transactions(state, observed)

    assert len(new_state.phase_d_buffer) == 2
    assert new_state.metadata["n_rows_phase_d_buffer"] == 2
    assert "last_observed_update_utc" in new_state.metadata
    assert set(new_state.phase_d_processed.index) == {201, 202}


def test_record_observed_transactions_does_not_mutate_input_state():
    state = _base_state()
    buffer_before = state.phase_d_buffer.copy()
    observed = pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0], "TransactionID": [203]})
    record_observed_transactions(state, observed)
    pd.testing.assert_frame_equal(state.phase_d_buffer, buffer_before)
    assert len(state.phase_d_processed) == 0


def test_record_observed_transactions_leaves_phase_f_untouched():
    state = _base_state()
    counts_before = state.phase_f_counts.copy()
    fraud_before = state.phase_f_global_fraud_count
    event_before = state.phase_f_global_event_count

    # An 'isFraud' column, if a caller mistakenly includes one, must simply
    # never be read -- Phase D's occurrence path never needs a label.
    observed = pd.DataFrame(
        {PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0], "TransactionID": [204], "isFraud": [1]}
    )
    new_state = record_observed_transactions(state, observed)

    pd.testing.assert_frame_equal(new_state.phase_f_counts, counts_before)
    assert new_state.phase_f_global_fraud_count == fraud_before
    assert new_state.phase_f_global_event_count == event_before
    assert len(new_state.phase_f_processed) == 0


def test_record_observed_transactions_works_without_isfraud():
    state = _base_state()
    observed = pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0], "TransactionID": [205]})
    new_state = record_observed_transactions(state, observed)
    assert len(new_state.phase_d_buffer) == 1


def test_record_observed_transactions_via_raw_key_components():
    config = load_config()
    state = _base_state()
    observed = pd.DataFrame(
        {
            **{col: [1] for col in config.payment_proxy_key_columns},
            "TransactionDT": [100],
            "TransactionAmt": [10.0],
            "TransactionID": [206],
        }
    )
    new_state = record_observed_transactions(state, observed, config=config)
    assert len(new_state.phase_d_buffer) == 1
    assert new_state.phase_d_buffer[PAYMENT_GROUP_COL].iloc[0] == "1|1|1|1|1"


def test_record_observed_transactions_requires_needed_columns():
    state = _base_state()
    observed = pd.DataFrame({PAYMENT_GROUP_COL: ["K1"]})  # missing TransactionDT/TransactionAmt/TransactionID
    with pytest.raises(ValueError):
        record_observed_transactions(state, observed)


def test_record_observed_transactions_requires_transaction_id():
    """TransactionID AVAILABILITY CONTRACT (explicit, dedicated proof): the
    state-MUTATING half of the contract -- unlike scoring (see
    tests/test_inference_scoring.py::test_score_transaction_succeeds_without_transaction_id),
    record_observed_transactions must fail CLEARLY, not silently proceed
    non-idempotently, when TransactionID is absent."""
    state = _base_state()
    observed = pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0]})  # no TransactionID
    with pytest.raises(ValueError):
        record_observed_transactions(state, observed)


def test_record_observed_transactions_update_resolved_labels_stay_separate():
    """update-resolved remains responsible only for Phase F label history,
    and record-observed remains responsible only for Phase D occurrence
    history -- neither path ever touches the other's state."""
    state = _base_state()

    after_record = record_observed_transactions(
        state, pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0], "TransactionID": [301]})
    )
    pd.testing.assert_frame_equal(after_record.phase_f_counts, state.phase_f_counts)
    assert after_record.phase_f_global_fraud_count == state.phase_f_global_fraud_count
    assert after_record.phase_f_global_event_count == state.phase_f_global_event_count
    assert len(after_record.phase_f_processed) == 0

    after_resolve = update_resolved_labels(
        state, pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "isFraud": [1], "TransactionID": [302]})
    )
    pd.testing.assert_frame_equal(after_resolve.phase_d_buffer, state.phase_d_buffer)
    assert len(after_resolve.phase_d_processed) == 0


def test_record_observed_transactions_does_not_see_itself():
    """Adversarial: a transaction, once recorded, must not appear in its
    own prior window even if scored again with the exact same
    (key, TransactionDT, TransactionAmt) -- Phase D's own same-timestamp
    bucket-tie rule (rows sharing (group, TransactionDT) never see each
    other) makes this true structurally, not by any special-casing in
    record_observed_transactions itself."""
    state = InferenceState(
        phase_d_buffer=_empty_d_buffer(),
        phase_f_counts=pd.DataFrame(columns=["fraud_count", "event_count"]).astype("int64"),
        phase_f_global_fraud_count=0,
        phase_f_global_event_count=0,
        metadata={},
    )
    txn = {PAYMENT_GROUP_COL: "K1", "TransactionDT": 100, "TransactionAmt": 10.0, "TransactionID": 401}
    new_state = record_observed_transactions(state, pd.DataFrame([txn]))
    assert len(new_state.phase_d_buffer) == 1

    # Re-score the SAME transaction against the post-record buffer.
    combined = pd.concat(
        [new_state.phase_d_buffer, pd.DataFrame([txn])[[PAYMENT_GROUP_COL, "TransactionDT", "TransactionAmt"]]],
        ignore_index=True,
    )
    scored = compute_behavioral_change_score(
        combined, _detection_config(), group_col=PAYMENT_GROUP_COL, amount_col="TransactionAmt", dt_col="TransactionDT"
    )
    assert int(scored.iloc[-1]["prior_count_in_window"]) == 0


def test_record_observed_transactions_same_timestamp_batch_does_not_cross_contaminate():
    """Adversarial batch-safety: two DIFFERENT transactions sharing the same
    TransactionDT, recorded together in one batch call, must not see each
    other when a third transaction queries at that same timestamp -- but
    both must be visible to a later transaction at a strictly later
    timestamp."""
    state = InferenceState(
        phase_d_buffer=_empty_d_buffer(),
        phase_f_counts=pd.DataFrame(columns=["fraud_count", "event_count"]).astype("int64"),
        phase_f_global_fraud_count=0,
        phase_f_global_event_count=0,
        metadata={},
    )
    observed = pd.DataFrame(
        {PAYMENT_GROUP_COL: ["K1", "K1"], "TransactionDT": [100, 100], "TransactionAmt": [10.0, 20.0], "TransactionID": [501, 502]}
    )
    new_state = record_observed_transactions(state, observed)
    assert len(new_state.phase_d_buffer) == 2

    detection_config = _detection_config()

    # A query at the SAME timestamp bucket must see neither prior row.
    same_dt_query = pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [999.0]})
    combined_same = pd.concat([new_state.phase_d_buffer, same_dt_query], ignore_index=True)
    scored_same = compute_behavioral_change_score(
        combined_same, detection_config, group_col=PAYMENT_GROUP_COL, amount_col="TransactionAmt", dt_col="TransactionDT"
    )
    assert int(scored_same.iloc[-1]["prior_count_in_window"]) == 0

    # A query at a strictly LATER timestamp must see both recorded rows.
    later_query = pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [200], "TransactionAmt": [999.0]})
    combined_later = pd.concat([new_state.phase_d_buffer, later_query], ignore_index=True)
    scored_later = compute_behavioral_change_score(
        combined_later, detection_config, group_col=PAYMENT_GROUP_COL, amount_col="TransactionAmt", dt_col="TransactionDT"
    )
    assert int(scored_later.iloc[-1]["prior_count_in_window"]) == 2


# ---------------------------------------------------------------------------
# TransactionID-keyed idempotency + CONFLICT POLICY -- adversarial proofs
# (see state.py's module docstring). Both update paths must be immune to:
# exact replay (same batch, separate calls, after save/load), a mixed
# new+already-processed batch, a duplicate ID within one batch, and must
# NOT confuse two distinct IDs sharing identical content. A conflicting
# resubmission (same ID, different payload) must raise, never silently
# apply. The two registries (Phase D observed vs. Phase F resolved) must
# stay independent.
# ---------------------------------------------------------------------------


def test_update_resolved_labels_exact_replay_is_idempotent():
    state = _base_state()
    resolved = pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "isFraud": [1], "TransactionID": [1001]})

    once = update_resolved_labels(state, resolved)
    twice = update_resolved_labels(once, resolved)

    assert once.phase_f_counts.loc["K1", "event_count"] == 4  # 3 + 1
    assert twice.phase_f_counts.loc["K1", "event_count"] == 4  # exact replay is a no-op
    assert twice.phase_f_counts.loc["K1", "fraud_count"] == once.phase_f_counts.loc["K1", "fraud_count"] == 2
    assert twice.phase_f_global_event_count == once.phase_f_global_event_count == 21
    assert twice.phase_f_global_fraud_count == once.phase_f_global_fraud_count == 6
    assert set(twice.phase_f_processed.index) == {1001}
    assert twice.metadata["n_duplicate_ids_skipped_last_resolved_update"] == 1


def test_record_observed_transactions_exact_replay_is_idempotent():
    state = _base_state()
    observed = pd.DataFrame(
        {PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0], "TransactionID": [2001]}
    )

    once = record_observed_transactions(state, observed)
    twice = record_observed_transactions(once, observed)

    assert len(once.phase_d_buffer) == 1
    assert len(twice.phase_d_buffer) == 1  # exact replay is a no-op
    assert set(twice.phase_d_processed.index) == {2001}
    assert twice.metadata["n_duplicate_ids_skipped_last_observed_update"] == 1


def test_record_observed_transactions_replay_after_save_load_is_idempotent(tmp_path):
    state = _base_state()
    observed = pd.DataFrame(
        {PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0], "TransactionID": [3001]}
    )
    once = record_observed_transactions(state, observed)
    save_state(once, tmp_path / "state")

    reloaded = load_state(tmp_path / "state")
    replayed = record_observed_transactions(reloaded, observed)

    # True no-op: buffer contents AND row count unchanged...
    assert len(reloaded.phase_d_buffer) == 1
    assert len(replayed.phase_d_buffer) == 1  # still a no-op after a save/load round-trip
    pd.testing.assert_frame_equal(replayed.phase_d_buffer, reloaded.phase_d_buffer)
    assert set(replayed.phase_d_processed.index) == {3001}
    # ...Phase F counters completely untouched (independent domain, and this
    # was a pure duplicate so nothing should move at all)...
    assert replayed.phase_f_global_fraud_count == reloaded.phase_f_global_fraud_count
    assert replayed.phase_f_global_event_count == reloaded.phase_f_global_event_count
    pd.testing.assert_frame_equal(replayed.phase_f_counts, reloaded.phase_f_counts)
    # ...and metadata correctly reflects a fully-skipped duplicate batch,
    # not a partial or silent success.
    assert replayed.metadata["n_rows_phase_d_buffer"] == reloaded.metadata["n_rows_phase_d_buffer"] == 1
    assert replayed.metadata["n_ids_phase_d_observed"] == reloaded.metadata["n_ids_phase_d_observed"] == 1
    assert replayed.metadata["n_duplicate_ids_skipped_last_observed_update"] == 1


def test_update_resolved_labels_replay_after_save_load_is_idempotent(tmp_path):
    state = _base_state()
    resolved = pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "isFraud": [1], "TransactionID": [3002]})
    once = update_resolved_labels(state, resolved)
    save_state(once, tmp_path / "state")

    reloaded = load_state(tmp_path / "state")
    replayed = update_resolved_labels(reloaded, resolved)

    # True no-op: counts AND global counters unchanged...
    assert reloaded.phase_f_counts.loc["K1", "event_count"] == 4
    assert replayed.phase_f_counts.loc["K1", "event_count"] == 4  # still a no-op after reload
    pd.testing.assert_frame_equal(replayed.phase_f_counts, reloaded.phase_f_counts)
    assert replayed.phase_f_global_fraud_count == reloaded.phase_f_global_fraud_count
    assert replayed.phase_f_global_event_count == reloaded.phase_f_global_event_count
    assert set(replayed.phase_f_processed.index) == {3002}
    # ...Phase D buffer completely untouched (independent domain)...
    pd.testing.assert_frame_equal(replayed.phase_d_buffer, reloaded.phase_d_buffer)
    # ...and metadata correctly reflects a fully-skipped duplicate batch.
    assert replayed.metadata["n_keys_phase_f"] == reloaded.metadata["n_keys_phase_f"]
    assert replayed.metadata["n_ids_phase_f_resolved"] == reloaded.metadata["n_ids_phase_f_resolved"] == 1
    assert replayed.metadata["n_duplicate_ids_skipped_last_resolved_update"] == 1


def test_record_observed_transactions_mixed_batch_new_and_processed_ids():
    state = _base_state()
    first = record_observed_transactions(
        state, pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0], "TransactionID": [4001]})
    )
    assert len(first.phase_d_buffer) == 1

    mixed = pd.DataFrame(
        {
            PAYMENT_GROUP_COL: ["K1", "K2"],
            "TransactionDT": [100, 200],
            "TransactionAmt": [10.0, 20.0],
            "TransactionID": [4001, 4002],  # 4001 already processed (identical payload), 4002 new
        }
    )
    second = record_observed_transactions(first, mixed)
    assert len(second.phase_d_buffer) == 2  # only the new ID (4002) was appended
    assert set(second.phase_d_processed.index) == {4001, 4002}
    assert second.metadata["n_duplicate_ids_skipped_last_observed_update"] == 1


def test_update_resolved_labels_mixed_batch_new_and_processed_ids():
    state = _base_state()
    first = update_resolved_labels(state, pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "isFraud": [1], "TransactionID": [5001]}))
    assert first.phase_f_counts.loc["K1", "event_count"] == 4
    assert first.phase_f_counts.loc["K1", "fraud_count"] == 2

    mixed = pd.DataFrame(
        {
            PAYMENT_GROUP_COL: ["K1", "K1"],
            "isFraud": [1, 0],
            "TransactionID": [5001, 5002],  # 5001 already resolved (identical payload), 5002 new
        }
    )
    second = update_resolved_labels(first, mixed)
    assert second.phase_f_counts.loc["K1", "event_count"] == 5  # only 5002 applied
    assert second.phase_f_counts.loc["K1", "fraud_count"] == 2  # 5002's isFraud=0 adds nothing
    assert set(second.phase_f_processed.index) == {5001, 5002}
    assert second.metadata["n_duplicate_ids_skipped_last_resolved_update"] == 1


def test_same_transaction_id_recorded_then_resolved_independent_domains():
    """A transaction can legitimately be recorded once (Phase D) and
    resolved later (Phase F) using the SAME TransactionID -- the two
    de-duplication domains must not block each other."""
    state = _base_state()
    txn_id = 6001
    after_record = record_observed_transactions(
        state, pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0], "TransactionID": [txn_id]})
    )
    assert len(after_record.phase_d_buffer) == 1
    assert len(after_record.phase_f_processed) == 0  # untouched by recording

    after_resolve = update_resolved_labels(
        after_record, pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "isFraud": [1], "TransactionID": [txn_id]})
    )
    # Resolving the SAME ID must NOT be blocked by it already being in
    # Phase D's (separate) registry, and must not conflict-check against it
    # either (the two registries carry different payload columns entirely).
    assert after_resolve.phase_f_counts.loc["K1", "event_count"] == 4
    assert set(after_resolve.phase_f_processed.index) == {txn_id}
    assert set(after_resolve.phase_d_processed.index) == {txn_id}  # untouched by resolving

    # And the reverse order: recording an occurrence must not be blocked by
    # that ID already being resolved in Phase F.
    state2 = _base_state()
    after_resolve2 = update_resolved_labels(
        state2, pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "isFraud": [1], "TransactionID": [txn_id]})
    )
    after_record2 = record_observed_transactions(
        after_resolve2,
        pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0], "TransactionID": [txn_id]}),
    )
    assert len(after_record2.phase_d_buffer) == 1  # not blocked by prior resolution of the same ID


def test_distinct_transaction_ids_with_identical_content_remain_distinct():
    """Dedup is keyed SOLELY on TransactionID -- two different transactions
    sharing the exact same (payment_proxy_key, TransactionDT,
    TransactionAmt) must both be kept, never collapsed by content."""
    state = _base_state()
    observed = pd.DataFrame(
        {
            PAYMENT_GROUP_COL: ["K1", "K1"],
            "TransactionDT": [100, 100],
            "TransactionAmt": [10.0, 10.0],
            "TransactionID": [7001, 7002],
        }
    )
    new_state = record_observed_transactions(state, observed)
    assert len(new_state.phase_d_buffer) == 2
    assert set(new_state.phase_d_processed.index) == {7001, 7002}


def test_record_observed_transactions_duplicate_id_within_same_batch_identical_payload():
    """Same TransactionID appearing twice in ONE batch with an IDENTICAL
    payload both times -- collapses to a single occurrence, no error."""
    state = _base_state()
    observed = pd.DataFrame(
        {
            PAYMENT_GROUP_COL: ["K1", "K1"],
            "TransactionDT": [100, 100],
            "TransactionAmt": [10.0, 10.0],
            "TransactionID": [8001, 8001],
        }
    )
    new_state = record_observed_transactions(state, observed)
    assert len(new_state.phase_d_buffer) == 1
    assert set(new_state.phase_d_processed.index) == {8001}
    assert new_state.metadata["n_duplicate_ids_skipped_last_observed_update"] == 1


def test_update_resolved_labels_duplicate_id_within_same_batch_identical_payload():
    state = _base_state()
    resolved = pd.DataFrame(
        {PAYMENT_GROUP_COL: ["K1", "K1"], "isFraud": [1, 1], "TransactionID": [8002, 8002]}
    )
    new_state = update_resolved_labels(state, resolved)
    assert new_state.phase_f_counts.loc["K1", "event_count"] == 4  # only ONE resolution applied
    assert new_state.phase_f_counts.loc["K1", "fraud_count"] == 2
    assert set(new_state.phase_f_processed.index) == {8002}


def test_record_observed_transactions_conflicting_payload_within_same_batch_raises():
    """Same TransactionID appearing twice in ONE batch with a DIFFERENT
    payload -- CONFLICT POLICY: raise, never silently keep the first or
    the last."""
    state = _base_state()
    observed = pd.DataFrame(
        {
            PAYMENT_GROUP_COL: ["K1", "K1"],
            "TransactionDT": [100, 200],  # different TransactionDT, SAME TransactionID
            "TransactionAmt": [10.0, 999.0],
            "TransactionID": [9001, 9001],
        }
    )
    with pytest.raises(ValueError, match="conflicting payload"):
        record_observed_transactions(state, observed)
    # Nothing should have been touched by the failed call (state is
    # immutable regardless, but confirm the ORIGINAL state's own buffer/
    # registry are untouched too).
    assert len(state.phase_d_buffer) == 0
    assert len(state.phase_d_processed) == 0


def test_update_resolved_labels_conflicting_payload_within_same_batch_raises():
    state = _base_state()
    resolved = pd.DataFrame(
        {
            PAYMENT_GROUP_COL: ["K1", "K2"],  # different payment_proxy_key, SAME TransactionID
            "isFraud": [1, 1],
            "TransactionID": [9002, 9002],
        }
    )
    with pytest.raises(ValueError, match="conflicting payload"):
        update_resolved_labels(state, resolved)


def test_record_observed_transactions_conflicting_payload_across_calls_raises():
    """The canonical CONFLICT POLICY case: a TransactionID already
    persisted in the registry (from a PRIOR call) is resubmitted with a
    DIFFERENT payload -- must raise, not silently overwrite or ignore."""
    state = _base_state()
    first = record_observed_transactions(
        state, pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0], "TransactionID": [9101]})
    )
    conflicting = pd.DataFrame(
        {PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [999.0], "TransactionID": [9101]}
    )
    with pytest.raises(ValueError, match="conflicting payload"):
        record_observed_transactions(first, conflicting)
    # The prior (valid) state must remain exactly as it was.
    assert len(first.phase_d_buffer) == 1
    assert float(first.phase_d_buffer["TransactionAmt"].iloc[0]) == 10.0


def test_update_resolved_labels_conflicting_payload_across_calls_raises():
    state = _base_state()
    first = update_resolved_labels(state, pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "isFraud": [1], "TransactionID": [9102]}))
    conflicting = pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "isFraud": [0], "TransactionID": [9102]})  # different label
    with pytest.raises(ValueError, match="conflicting payload"):
        update_resolved_labels(first, conflicting)
    assert first.phase_f_counts.loc["K1", "fraud_count"] == 2
    assert first.phase_f_counts.loc["K1", "event_count"] == 4


def test_record_observed_transactions_conflicting_payload_after_save_load_raises(tmp_path):
    state = _base_state()
    first = record_observed_transactions(
        state, pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0], "TransactionID": [9201]})
    )
    save_state(first, tmp_path / "state")
    reloaded = load_state(tmp_path / "state")

    conflicting = pd.DataFrame(
        {PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [999], "TransactionAmt": [10.0], "TransactionID": [9201]}
    )
    with pytest.raises(ValueError, match="conflicting payload"):
        record_observed_transactions(reloaded, conflicting)


def test_load_state_backward_compatible_with_pre_idempotency_directory(tmp_path):
    """A state directory saved by the PRE-idempotency version of this
    module never wrote phase_d_processed.parquet/phase_f_processed.parquet
    -- load_state must treat their absence as 'no IDs tracked yet', not an
    error, and the loaded state must remain fully usable afterward."""
    state = _base_state()
    save_state(state, tmp_path / "state")

    (tmp_path / "state" / "phase_d_processed.parquet").unlink()
    (tmp_path / "state" / "phase_f_processed.parquet").unlink()

    loaded = load_state(tmp_path / "state")
    assert len(loaded.phase_d_processed) == 0
    assert len(loaded.phase_f_processed) == 0

    after_record = record_observed_transactions(
        loaded, pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0], "TransactionID": [9301]})
    )
    assert len(after_record.phase_d_buffer) == 1
    assert set(after_record.phase_d_processed.index) == {9301}


def test_record_observed_transactions_full_duplicate_batch_is_pure_noop_with_correct_metadata():
    """Empty/no-op replay: a batch where EVERY TransactionID is already
    processed must leave the buffer entirely unchanged and report the
    correct duplicate-skip count in metadata (exercises the empty
    'new_rows' path through groupby/concat without error)."""
    state = _base_state()
    first = record_observed_transactions(
        state,
        pd.DataFrame(
            {PAYMENT_GROUP_COL: ["K1", "K2"], "TransactionDT": [100, 200], "TransactionAmt": [10.0, 20.0], "TransactionID": [9401, 9402]}
        ),
    )
    assert len(first.phase_d_buffer) == 2

    replay = record_observed_transactions(
        first,
        pd.DataFrame(
            {PAYMENT_GROUP_COL: ["K1", "K2"], "TransactionDT": [100, 200], "TransactionAmt": [10.0, 20.0], "TransactionID": [9401, 9402]}
        ),
    )
    assert len(replay.phase_d_buffer) == 2  # unchanged
    pd.testing.assert_frame_equal(replay.phase_d_buffer, first.phase_d_buffer)
    assert replay.metadata["n_rows_phase_d_buffer"] == 2
    assert replay.metadata["n_duplicate_ids_skipped_last_observed_update"] == 2
    assert replay.metadata["n_ids_phase_d_observed"] == 2


def test_update_resolved_labels_full_duplicate_batch_is_pure_noop_with_correct_metadata():
    state = _base_state()
    first = update_resolved_labels(
        state, pd.DataFrame({PAYMENT_GROUP_COL: ["K1", "K2"], "isFraud": [1, 0], "TransactionID": [9501, 9502]})
    )
    replay = update_resolved_labels(
        first, pd.DataFrame({PAYMENT_GROUP_COL: ["K1", "K2"], "isFraud": [1, 0], "TransactionID": [9501, 9502]})
    )
    pd.testing.assert_frame_equal(replay.phase_f_counts, first.phase_f_counts)
    assert replay.phase_f_global_event_count == first.phase_f_global_event_count
    assert replay.phase_f_global_fraud_count == first.phase_f_global_fraud_count
    assert replay.metadata["n_duplicate_ids_skipped_last_resolved_update"] == 2
    assert replay.metadata["n_ids_phase_f_resolved"] == 2


# ---------------------------------------------------------------------------
# Persistence audit -- save/load round-trip after record-observed and after
# update-resolved (individually and combined), including a multi-batch
# append that actually exercises record_observed_transactions's pd.concat
# branch (not just its empty-buffer fast path).
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_after_record_observed(tmp_path):
    state = _base_state()
    s1 = record_observed_transactions(
        state, pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0], "TransactionID": [1101]})
    )
    # A second batch, appended onto a now-nonempty buffer -- exercises the
    # pd.concat branch, not record_observed_transactions's empty-buffer
    # fast path (test_save_load_state_roundtrip only covers the latter).
    s2 = record_observed_transactions(
        s1,
        pd.DataFrame(
            {
                PAYMENT_GROUP_COL: ["K1", "K2"],
                "TransactionDT": [200, 300],
                "TransactionAmt": [20.0, 30.0],
                "TransactionID": [1102, 1103],
            }
        ),
    )
    assert len(s2.phase_d_buffer) == 3

    save_state(s2, tmp_path / "state")
    loaded = load_state(tmp_path / "state")

    pd.testing.assert_frame_equal(loaded.phase_d_buffer, s2.phase_d_buffer)
    pd.testing.assert_frame_equal(loaded.phase_f_counts, s2.phase_f_counts)
    assert loaded.phase_f_global_fraud_count == s2.phase_f_global_fraud_count
    assert loaded.phase_f_global_event_count == s2.phase_f_global_event_count
    assert set(loaded.phase_d_processed.index) == set(s2.phase_d_processed.index) == {1101, 1102, 1103}
    assert loaded.metadata["last_observed_update_utc"] == s2.metadata["last_observed_update_utc"]
    assert loaded.metadata["n_rows_phase_d_buffer"] == 3


def test_save_load_roundtrip_after_update_resolved(tmp_path):
    state = _base_state()
    s1 = update_resolved_labels(
        state, pd.DataFrame({PAYMENT_GROUP_COL: ["K1", "K2"], "isFraud": [1, 0], "TransactionID": [1201, 1202]})
    )

    save_state(s1, tmp_path / "state")
    loaded = load_state(tmp_path / "state")

    pd.testing.assert_frame_equal(loaded.phase_f_counts, s1.phase_f_counts)
    pd.testing.assert_frame_equal(loaded.phase_d_buffer, s1.phase_d_buffer)
    assert loaded.phase_f_global_fraud_count == s1.phase_f_global_fraud_count
    assert loaded.phase_f_global_event_count == s1.phase_f_global_event_count
    assert set(loaded.phase_f_processed.index) == set(s1.phase_f_processed.index) == {1201, 1202}
    assert loaded.metadata["last_resolved_update_utc"] == s1.metadata["last_resolved_update_utc"]


def test_save_load_roundtrip_after_record_observed_and_update_resolved_combined(tmp_path):
    state = _base_state()
    after_record = record_observed_transactions(
        state,
        pd.DataFrame(
            {PAYMENT_GROUP_COL: ["K1", "K2"], "TransactionDT": [100, 200], "TransactionAmt": [10.0, 20.0], "TransactionID": [1301, 1302]}
        ),
    )
    after_both = update_resolved_labels(
        after_record, pd.DataFrame({PAYMENT_GROUP_COL: ["K2"], "isFraud": [1], "TransactionID": [1401]})
    )

    save_state(after_both, tmp_path / "state")
    loaded = load_state(tmp_path / "state")

    pd.testing.assert_frame_equal(loaded.phase_d_buffer, after_both.phase_d_buffer)
    pd.testing.assert_frame_equal(loaded.phase_f_counts, after_both.phase_f_counts)
    assert loaded.phase_f_global_fraud_count == after_both.phase_f_global_fraud_count
    assert loaded.phase_f_global_event_count == after_both.phase_f_global_event_count
    assert set(loaded.phase_d_processed.index) == set(after_both.phase_d_processed.index) == {1301, 1302}
    assert set(loaded.phase_f_processed.index) == set(after_both.phase_f_processed.index) == {1401}
    assert loaded.metadata["last_observed_update_utc"] == after_both.metadata["last_observed_update_utc"]
    assert loaded.metadata["last_resolved_update_utc"] == after_both.metadata["last_resolved_update_utc"]

    # The loaded state must still be USABLE for a real Phase D query -- not
    # merely byte-equal -- confirming parquet round-trip preserved dtypes
    # (group/int64 TransactionDT/float64 TransactionAmt) well enough for
    # compute_behavioral_change_score to run against it without error.
    later_query = pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [500], "TransactionAmt": [15.0]})
    combined = pd.concat([loaded.phase_d_buffer, later_query], ignore_index=True)
    scored = compute_behavioral_change_score(
        combined, _detection_config(), group_col=PAYMENT_GROUP_COL, amount_col="TransactionAmt", dt_col="TransactionDT"
    )
    assert int(scored.iloc[-1]["prior_count_in_window"]) == 1

    # And idempotency itself must survive the round-trip: replaying either
    # already-processed ID against the RELOADED state must remain a no-op.
    replay_record = record_observed_transactions(
        loaded,
        pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [100], "TransactionAmt": [10.0], "TransactionID": [1301]}),
    )
    assert len(replay_record.phase_d_buffer) == len(loaded.phase_d_buffer)
    replay_resolve = update_resolved_labels(
        loaded, pd.DataFrame({PAYMENT_GROUP_COL: ["K2"], "isFraud": [1], "TransactionID": [1401]})
    )
    assert replay_resolve.phase_f_counts.loc["K2", "event_count"] == loaded.phase_f_counts.loc["K2", "event_count"]

    # And conflict detection also survives the round-trip.
    with pytest.raises(ValueError, match="conflicting payload"):
        record_observed_transactions(
            loaded,
            pd.DataFrame({PAYMENT_GROUP_COL: ["K1"], "TransactionDT": [999], "TransactionAmt": [10.0], "TransactionID": [1301]}),
        )

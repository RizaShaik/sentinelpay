import json

import numpy as np
import pandas as pd

import sentinelpay.eda.run_phase_g as run_phase_g_module
import sentinelpay.eda.run_phase_h as run_phase_h_module
import sentinelpay.inference.artifacts as artifacts_module
from sentinelpay.config import load_config
from sentinelpay.inference.cli import main
from sentinelpay.inference.state import load_state
from sentinelpay.model_features import LADDER_FEATURE_COLUMNS


def _synthetic_frame_for(config, days):
    seconds_per_day = config.seconds_per_day
    rng = np.random.default_rng(0)
    rows = []
    for partition, day in days.items():
        for i in range(20):
            rows.append(
                {
                    "TransactionID": len(rows) + 1,
                    "TransactionDT": day * seconds_per_day + i,
                    "TransactionAmt": float(10 + i),
                    "isFraud": int(rng.random() < 0.2),
                    **{col: int(i % 4) for col in config.payment_proxy_key_columns},
                }
            )
    return pd.DataFrame(rows)


def _patch_all_loaders(monkeypatch, config):
    dev_days = {"train": 5, "embargo_1": 133, "validation": 150, "embargo_2": 165}
    full_days = {**dev_days, "holdout": 175}
    dev_frame = _synthetic_frame_for(config, dev_days)
    full_frame = _synthetic_frame_for(config, full_days)

    def fake_dev_loader(split, columns, config=None):
        return dev_frame[columns].copy()

    def fake_full_loader(split, columns, config=None):
        return full_frame[columns].copy()

    def fake_identity_ids(split, config=None):
        return pd.Series([], dtype="int64")

    monkeypatch.setattr(run_phase_g_module, "load_transaction_columns", fake_dev_loader)
    monkeypatch.setattr(run_phase_h_module, "load_transaction_columns", fake_full_loader)
    monkeypatch.setattr(artifacts_module, "load_identity_ids", fake_identity_ids)


def test_cli_build_score_update_resolved_end_to_end(monkeypatch, tmp_path, capsys):
    config = load_config()
    _patch_all_loaders(monkeypatch, config)

    artifact_path = tmp_path / "model.joblib"
    state_dir = tmp_path / "state"

    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "build"])
    assert artifact_path.exists()
    assert (state_dir / "phase_d_buffer.parquet").exists()
    assert (state_dir / "phase_f_counts.parquet").exists()
    assert (state_dir / "state_meta.json").exists()
    build_output = capsys.readouterr().out
    build_meta = json.loads(build_output)
    assert build_meta["model_metadata"]["holdout_used_for_fitting"] is False
    assert "resolution-only" in build_meta["state_metadata"]["phase_f_update_policy"]

    txn = {
        "TransactionID": 999_999_999,
        "TransactionDT": 999_999,
        "TransactionAmt": 42.0,
        "has_identity": 1,
        **{col: 1 for col in config.payment_proxy_key_columns},
    }
    txn_path = tmp_path / "txn.json"
    txn_path.write_text(json.dumps(txn), encoding="utf-8")

    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "score", "--input", str(txn_path)])
    score_output = json.loads(capsys.readouterr().out)
    assert 0.0 <= score_output["fraud_probability"] <= 1.0
    # Real F2 feature-contract check -- not just "isFraud absent" (which is
    # trivially true since LADDER_FEATURE_COLUMNS never names it): the CLI's
    # emitted feature set must be EXACTLY the frozen F2 column list, in the
    # frozen order, matching sentinelpay.model_features.LADDER_FEATURE_COLUMNS.
    assert list(score_output["features"].keys()) == LADDER_FEATURE_COLUMNS["F2"]
    assert "isFraud" not in score_output["features"]
    assert "payment_proxy_key" in score_output
    assert "phase_d_diagnostics" in score_output and "phase_f_diagnostics" in score_output

    state_before_record = load_state(state_dir)
    phase_f_counts_before = state_before_record.phase_f_counts.copy()

    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "record-observed", "--input", str(txn_path)])
    capsys.readouterr()

    state_after_record = load_state(state_dir)
    assert len(state_after_record.phase_d_buffer) == len(state_before_record.phase_d_buffer) + 1
    pd.testing.assert_frame_equal(state_after_record.phase_f_counts, phase_f_counts_before)
    assert state_after_record.phase_f_global_event_count == state_before_record.phase_f_global_event_count
    assert "last_observed_update_utc" in state_after_record.metadata

    # record-observed is idempotent -- replaying the SAME txn.json (same
    # TransactionID) must be a no-op, not a second append.
    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "record-observed", "--input", str(txn_path)])
    capsys.readouterr()
    state_after_replay = load_state(state_dir)
    assert len(state_after_replay.phase_d_buffer) == len(state_after_record.phase_d_buffer)

    resolved_path = tmp_path / "resolved.json"
    resolved_payload = [{**{col: 1 for col in config.payment_proxy_key_columns}, "isFraud": 1, "TransactionID": 888_888_888}]
    resolved_path.write_text(json.dumps(resolved_payload), encoding="utf-8")
    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "update-resolved", "--input", str(resolved_path)])
    capsys.readouterr()

    meta_after = json.loads((state_dir / "state_meta.json").read_text(encoding="utf-8"))
    assert "last_resolved_update_utc" in meta_after

    state_after_resolve = load_state(state_dir)
    resolved_key = "1|1|1|1|1"
    fraud_count_after_first_resolve = int(state_after_resolve.phase_f_counts.loc[resolved_key, "fraud_count"])

    # update-resolved is idempotent -- replaying the SAME resolved.json
    # (same TransactionID) must not increment the counter a second time.
    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "update-resolved", "--input", str(resolved_path)])
    capsys.readouterr()
    state_after_resolve_replay = load_state(state_dir)
    assert int(state_after_resolve_replay.phase_f_counts.loc[resolved_key, "fraud_count"]) == fraud_count_after_first_resolve


def _built_cli_fixture(monkeypatch, tmp_path):
    """Shared setup for the duplicate-outcome CLI reporting tests: builds a
    fresh artifact/state directory and returns (config, artifact_path, state_dir)."""
    config = load_config()
    _patch_all_loaders(monkeypatch, config)
    artifact_path = tmp_path / "model.joblib"
    state_dir = tmp_path / "state"
    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "build"])
    return config, artifact_path, state_dir


def test_record_observed_all_new_batch_reports_all_recorded(monkeypatch, tmp_path, capsys):
    config, artifact_path, state_dir = _built_cli_fixture(monkeypatch, tmp_path)

    records = [
        {
            "TransactionID": 700_000_000 + i,
            "TransactionDT": 900_000 + i,
            "TransactionAmt": float(10 + i),
            **{col: 1 for col in config.payment_proxy_key_columns},
        }
        for i in range(3)
    ]
    input_path = tmp_path / "observed_all_new.json"
    input_path.write_text(json.dumps(records), encoding="utf-8")

    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "record-observed", "--input", str(input_path)])
    stderr = capsys.readouterr().err
    assert "Processed 3 observed transaction(s)" in stderr
    assert "3 newly recorded" in stderr
    assert "0 duplicate(s) skipped" in stderr


def test_record_observed_all_duplicate_batch_reports_zero_new(monkeypatch, tmp_path, capsys):
    config, artifact_path, state_dir = _built_cli_fixture(monkeypatch, tmp_path)

    records = [
        {
            "TransactionID": 700_100_000 + i,
            "TransactionDT": 910_000 + i,
            "TransactionAmt": float(20 + i),
            **{col: 1 for col in config.payment_proxy_key_columns},
        }
        for i in range(3)
    ]
    input_path = tmp_path / "observed_dupe.json"
    input_path.write_text(json.dumps(records), encoding="utf-8")

    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "record-observed", "--input", str(input_path)])
    capsys.readouterr()
    state_after_first = load_state(state_dir)

    # Replay the exact same batch -- every row is now a duplicate.
    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "record-observed", "--input", str(input_path)])
    stderr = capsys.readouterr().err
    assert "Processed 3 observed transaction(s)" in stderr
    assert "0 newly recorded" in stderr
    assert "3 duplicate(s) skipped" in stderr

    state_after_replay = load_state(state_dir)
    assert len(state_after_replay.phase_d_buffer) == len(state_after_first.phase_d_buffer)


def test_record_observed_mixed_batch_reports_accepted_and_duplicates(monkeypatch, tmp_path, capsys):
    config, artifact_path, state_dir = _built_cli_fixture(monkeypatch, tmp_path)

    first_records = [
        {
            "TransactionID": 700_200_000 + i,
            "TransactionDT": 920_000 + i,
            "TransactionAmt": float(30 + i),
            **{col: 1 for col in config.payment_proxy_key_columns},
        }
        for i in range(2)
    ]
    first_path = tmp_path / "observed_first.json"
    first_path.write_text(json.dumps(first_records), encoding="utf-8")
    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "record-observed", "--input", str(first_path)])
    capsys.readouterr()
    state_before_mixed = load_state(state_dir)

    # Mixed batch: the 2 rows above (now duplicates) + 2 genuinely new rows.
    new_records = [
        {
            "TransactionID": 700_300_000 + i,
            "TransactionDT": 930_000 + i,
            "TransactionAmt": float(40 + i),
            **{col: 1 for col in config.payment_proxy_key_columns},
        }
        for i in range(2)
    ]
    mixed_records = first_records + new_records
    mixed_path = tmp_path / "observed_mixed.json"
    mixed_path.write_text(json.dumps(mixed_records), encoding="utf-8")

    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "record-observed", "--input", str(mixed_path)])
    stderr = capsys.readouterr().err
    assert "Processed 4 observed transaction(s)" in stderr
    assert "2 newly recorded" in stderr
    assert "2 duplicate(s) skipped" in stderr

    state_after_mixed = load_state(state_dir)
    assert len(state_after_mixed.phase_d_buffer) == len(state_before_mixed.phase_d_buffer) + 2


def test_update_resolved_all_new_batch_reports_all_updated(monkeypatch, tmp_path, capsys):
    config, artifact_path, state_dir = _built_cli_fixture(monkeypatch, tmp_path)

    records = [
        {**{col: 1 for col in config.payment_proxy_key_columns}, "isFraud": 0, "TransactionID": 800_000_000 + i}
        for i in range(3)
    ]
    input_path = tmp_path / "resolved_all_new.json"
    input_path.write_text(json.dumps(records), encoding="utf-8")

    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "update-resolved", "--input", str(input_path)])
    stderr = capsys.readouterr().err
    assert "Processed 3 resolved label(s)" in stderr
    assert "3 newly updated" in stderr
    assert "0 duplicate(s) skipped" in stderr


def test_update_resolved_all_duplicate_batch_reports_zero_new(monkeypatch, tmp_path, capsys):
    config, artifact_path, state_dir = _built_cli_fixture(monkeypatch, tmp_path)

    records = [
        {**{col: 1 for col in config.payment_proxy_key_columns}, "isFraud": 1, "TransactionID": 800_100_000 + i}
        for i in range(3)
    ]
    input_path = tmp_path / "resolved_dupe.json"
    input_path.write_text(json.dumps(records), encoding="utf-8")

    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "update-resolved", "--input", str(input_path)])
    capsys.readouterr()
    state_after_first = load_state(state_dir)

    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "update-resolved", "--input", str(input_path)])
    stderr = capsys.readouterr().err
    assert "Processed 3 resolved label(s)" in stderr
    assert "0 newly updated" in stderr
    assert "3 duplicate(s) skipped" in stderr

    state_after_replay = load_state(state_dir)
    pd.testing.assert_frame_equal(state_after_replay.phase_f_counts, state_after_first.phase_f_counts)


def test_update_resolved_mixed_batch_reports_accepted_and_duplicates(monkeypatch, tmp_path, capsys):
    config, artifact_path, state_dir = _built_cli_fixture(monkeypatch, tmp_path)

    first_records = [
        {**{col: 1 for col in config.payment_proxy_key_columns}, "isFraud": 1, "TransactionID": 800_200_000 + i}
        for i in range(2)
    ]
    first_path = tmp_path / "resolved_first.json"
    first_path.write_text(json.dumps(first_records), encoding="utf-8")
    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "update-resolved", "--input", str(first_path)])
    capsys.readouterr()

    new_records = [
        {**{col: 1 for col in config.payment_proxy_key_columns}, "isFraud": 0, "TransactionID": 800_300_000 + i}
        for i in range(2)
    ]
    mixed_records = first_records + new_records
    mixed_path = tmp_path / "resolved_mixed.json"
    mixed_path.write_text(json.dumps(mixed_records), encoding="utf-8")

    main(["--artifact-path", str(artifact_path), "--state-dir", str(state_dir), "update-resolved", "--input", str(mixed_path)])
    stderr = capsys.readouterr().err
    assert "Processed 4 resolved label(s)" in stderr
    assert "2 newly updated" in stderr
    assert "2 duplicate(s) skipped" in stderr

import numpy as np
import pandas as pd

import sentinelpay.eda.run_phase_g as run_phase_g_module
import sentinelpay.eda.run_phase_h as run_phase_h_module
import sentinelpay.inference.artifacts as artifacts_module
from sentinelpay.config import load_config
from sentinelpay.inference.state import build_initial_state, load_state, save_state
from ui_state import (
    ensure_scratch_state,
    record_observed,
    reset_scratch_state,
    resolve_transaction,
    state_summary,
)


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


def _build_synthetic_canonical(monkeypatch, tmp_path):
    """Builds a small synthetic state directory standing in for the real
    canonical `artifacts/inference/state/` -- keeps these tests independent
    of the real, large, generated artifacts."""
    config = load_config()
    _patch_all_loaders(monkeypatch, config)
    canonical_dir = tmp_path / "canonical_state"
    save_state(build_initial_state(), canonical_dir)
    return config, canonical_dir


def _hash_dir(dir_path):
    return {
        p.name: p.read_bytes() for p in sorted(dir_path.iterdir())
    }


def test_ensure_scratch_state_copies_and_is_idempotent(monkeypatch, tmp_path):
    _config, canonical_dir = _build_synthetic_canonical(monkeypatch, tmp_path)
    scratch_dir = tmp_path / "scratch_state"

    ensure_scratch_state(canonical_dir, scratch_dir)
    assert scratch_dir.exists()
    scratch_state = load_state(scratch_dir)
    canonical_state = load_state(canonical_dir)
    assert len(scratch_state.phase_d_buffer) == len(canonical_state.phase_d_buffer)

    # Mutate scratch directly, then call ensure_scratch_state again -- an
    # already-existing scratch directory must be left alone (no re-copy).
    (scratch_dir / "sentinel_marker.txt").write_text("mutated", encoding="utf-8")
    ensure_scratch_state(canonical_dir, scratch_dir)
    assert (scratch_dir / "sentinel_marker.txt").exists()


def test_reset_scratch_state_restores_from_canonical_without_touching_canonical(monkeypatch, tmp_path):
    config, canonical_dir = _build_synthetic_canonical(monkeypatch, tmp_path)
    scratch_dir = tmp_path / "scratch_state"
    ensure_scratch_state(canonical_dir, scratch_dir)

    canonical_before = _hash_dir(canonical_dir)

    scratch_state = load_state(scratch_dir)
    key_cols = config.payment_proxy_key_columns
    new_row = {
        "TransactionID": 555_555_555,
        "TransactionDT": 99_999_999,
        "TransactionAmt": 1.0,
        **{col: 7 for col in key_cols},
    }
    mutated_state, n_new, n_dup = record_observed(scratch_state, [new_row], scratch_dir)
    assert (n_new, n_dup) == (1, 0)
    assert len(mutated_state.phase_d_buffer) == len(scratch_state.phase_d_buffer) + 1

    reset_scratch_state(canonical_dir, scratch_dir)
    reset_state = load_state(scratch_dir)
    canonical_state = load_state(canonical_dir)
    assert len(reset_state.phase_d_buffer) == len(canonical_state.phase_d_buffer)
    assert 555_555_555 not in reset_state.phase_d_processed.index

    canonical_after = _hash_dir(canonical_dir)
    assert canonical_before == canonical_after


def test_record_observed_and_resolve_never_touch_canonical(monkeypatch, tmp_path):
    config, canonical_dir = _build_synthetic_canonical(monkeypatch, tmp_path)
    scratch_dir = tmp_path / "scratch_state"
    ensure_scratch_state(canonical_dir, scratch_dir)
    canonical_before = _hash_dir(canonical_dir)

    key_cols = config.payment_proxy_key_columns
    state = load_state(scratch_dir)
    observed_row = {
        "TransactionID": 111_222_333,
        "TransactionDT": 88_888_888,
        "TransactionAmt": 5.0,
        **{col: 3 for col in key_cols},
    }
    state, n_new, n_dup = record_observed(state, [observed_row], scratch_dir)
    assert (n_new, n_dup) == (1, 0)

    resolved_row = {"TransactionID": 111_222_444, "isFraud": 1, **{col: 3 for col in key_cols}}
    state, n_new, n_dup = resolve_transaction(state, [resolved_row], scratch_dir)
    assert (n_new, n_dup) == (1, 0)

    canonical_after = _hash_dir(canonical_dir)
    assert canonical_before == canonical_after


def test_record_observed_duplicate_reporting(monkeypatch, tmp_path):
    config, canonical_dir = _build_synthetic_canonical(monkeypatch, tmp_path)
    scratch_dir = tmp_path / "scratch_state"
    ensure_scratch_state(canonical_dir, scratch_dir)
    state = load_state(scratch_dir)

    key_cols = config.payment_proxy_key_columns
    row = {
        "TransactionID": 222_333_444,
        "TransactionDT": 77_777_777,
        "TransactionAmt": 9.0,
        **{col: 2 for col in key_cols},
    }
    state, n_new, n_dup = record_observed(state, [row], scratch_dir)
    assert (n_new, n_dup) == (1, 0)

    state, n_new, n_dup = record_observed(state, [row], scratch_dir)
    assert (n_new, n_dup) == (0, 1)


def test_resolve_transaction_duplicate_reporting(monkeypatch, tmp_path):
    config, canonical_dir = _build_synthetic_canonical(monkeypatch, tmp_path)
    scratch_dir = tmp_path / "scratch_state"
    ensure_scratch_state(canonical_dir, scratch_dir)
    state = load_state(scratch_dir)

    key_cols = config.payment_proxy_key_columns
    row = {"TransactionID": 333_444_555, "isFraud": 0, **{col: 5 for col in key_cols}}
    state, n_new, n_dup = resolve_transaction(state, [row], scratch_dir)
    assert (n_new, n_dup) == (1, 0)

    state, n_new, n_dup = resolve_transaction(state, [row], scratch_dir)
    assert (n_new, n_dup) == (0, 1)


def test_state_summary_matches_state_attributes(monkeypatch, tmp_path):
    _config, canonical_dir = _build_synthetic_canonical(monkeypatch, tmp_path)
    state = load_state(canonical_dir)
    summary = state_summary(state)
    assert summary == {
        "phase_d_buffer_size": len(state.phase_d_buffer),
        "phase_d_processed_ids": len(state.phase_d_processed),
        "phase_f_keys": len(state.phase_f_counts),
        "phase_f_processed_ids": len(state.phase_f_processed),
        "global_fraud_count": state.phase_f_global_fraud_count,
        "global_event_count": state.phase_f_global_event_count,
    }

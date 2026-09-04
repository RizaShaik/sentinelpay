"""Streamlit demo support: scratch-state management and thin presentation
wrappers around the real Phase I backend (`sentinelpay.inference.state`).

No fraud logic lives here. Every number this module produces is either a
plain `len()`/attribute read off a real `InferenceState`, or the direct
return value of `record_observed_transactions`/`update_resolved_labels`,
called completely unmodified. This module exists only so `app.py`'s
Streamlit callbacks and this module's own pytest tests can share the same
scratch-directory and duplicate-counting logic without re-deriving it in
either place.

`artifacts/inference/state/` (CANONICAL_STATE_DIR) is the real submission
snapshot and is treated as READ-ONLY here -- every function below either
only reads it or copies FROM it, never writes INTO it. All demo mutations
(`record_observed`, `resolve_transaction`) are saved to a separate scratch
directory (SCRATCH_STATE_DIR) instead.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from sentinelpay.inference.state import (
    InferenceState,
    load_state,
    record_observed_transactions,
    save_state,
    update_resolved_labels,
)

CANONICAL_STATE_DIR = Path("artifacts/inference/state")
CANONICAL_ARTIFACT_PATH = Path("artifacts/inference/model.joblib")
SCRATCH_STATE_DIR = Path("artifacts/inference/_demo_scratch_state")


def ensure_scratch_state(
    canonical_dir: Path = CANONICAL_STATE_DIR, scratch_dir: Path = SCRATCH_STATE_DIR
) -> None:
    """Creates `scratch_dir` as a copy of `canonical_dir` if it does not
    already exist. A no-op if the scratch directory is already present
    (so an app restart resumes the existing demo session rather than
    silently discarding it). Never writes to `canonical_dir`."""
    if not scratch_dir.exists():
        shutil.copytree(canonical_dir, scratch_dir)


def reset_scratch_state(
    canonical_dir: Path = CANONICAL_STATE_DIR, scratch_dir: Path = SCRATCH_STATE_DIR
) -> None:
    """Discards any demo mutations by replacing `scratch_dir` with a fresh
    copy of `canonical_dir`. Never writes to `canonical_dir`."""
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    shutil.copytree(canonical_dir, scratch_dir)


def load_scratch_state(scratch_dir: Path = SCRATCH_STATE_DIR) -> InferenceState:
    return load_state(scratch_dir)


def state_summary(state: InferenceState) -> dict:
    """Read-only summary metrics for display -- plain attribute/len reads
    off the real `InferenceState`, no derived fraud logic."""
    return {
        "phase_d_buffer_size": len(state.phase_d_buffer),
        "phase_d_processed_ids": len(state.phase_d_processed),
        "phase_f_keys": len(state.phase_f_counts),
        "phase_f_processed_ids": len(state.phase_f_processed),
        "global_fraud_count": state.phase_f_global_fraud_count,
        "global_event_count": state.phase_f_global_event_count,
    }


def record_observed(
    state: InferenceState, records: list[dict], scratch_dir: Path = SCRATCH_STATE_DIR
) -> tuple[InferenceState, int, int]:
    """Calls `record_observed_transactions` + `save_state` (both
    unmodified) against `scratch_dir`. Returns `(new_state, n_new,
    n_duplicates)`, derived the same way `cli.py`'s `cmd_record_observed`
    derives them: from `new_state.metadata`, not from re-inspecting the
    buffer."""
    observed_df = pd.DataFrame(records)
    new_state = record_observed_transactions(state, observed_df)
    save_state(new_state, scratch_dir)
    n_duplicates = int(new_state.metadata["n_duplicate_ids_skipped_last_observed_update"])
    n_new = len(records) - n_duplicates
    return new_state, n_new, n_duplicates


def resolve_transaction(
    state: InferenceState, records: list[dict], scratch_dir: Path = SCRATCH_STATE_DIR
) -> tuple[InferenceState, int, int]:
    """Calls `update_resolved_labels` + `save_state` (both unmodified)
    against `scratch_dir`. Returns `(new_state, n_new, n_duplicates)`, same
    derivation pattern as `record_observed` above."""
    resolved_df = pd.DataFrame(records)
    new_state = update_resolved_labels(state, resolved_df)
    save_state(new_state, scratch_dir)
    n_duplicates = int(new_state.metadata["n_duplicate_ids_skipped_last_resolved_update"])
    n_new = len(records) - n_duplicates
    return new_state, n_new, n_duplicates

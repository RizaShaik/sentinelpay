"""Phase I inference state: the per-`payment_proxy_key` (and global) running
totals a live scoring system needs to reproduce Phase D's and Phase F's
proven features WITHOUT re-scanning the full historical dataset on every
score.

`sentinelpay.target_history` and `sentinelpay.model_features` are imported
from, never modified. `sentinelpay.eda.run_phase_h.build_full_frame` is
imported (a pure data loader) to load all five partitions the same way
Phase H already does -- this module does not rerun Phase H's evaluation and
never touches `reports/eda/phase_h_results.json`.

===========================================================================
STATE SOURCE CONTRACT -- distinguishes Phase D's and Phase F's state
===========================================================================

**Phase D state** (`phase_d_buffer`): the raw `(TransactionDT, TransactionAmt)`
history per key, drawn from ALL FIVE partitions (`train`, `embargo_1`,
`validation`, `embargo_2`, `holdout`) -- UNRESTRICTED, exactly matching
Phase D's own already-approved non-target semantics (embargo content is a
legitimate source; Phase D never reads `isFraud` at all, so there is no
"resolution" concept for it -- see `sentinelpay.inference.scoring` for how
this buffer feeds `sentinelpay.detection.compute_behavioral_change_score`,
called unmodified, per new transaction).

**Phase F state** (`phase_f_counts`, `phase_f_global_*`): per-key and global
`(fraud_count, event_count)`, drawn ONLY from `train` + `validation` +
`holdout` -- `embargo_1`/`embargo_2` EXCLUDED, exactly matching Phase F's
frozen target-history contract.

**Initial cutoff**: `train` + `validation` + `holdout`, ALL resolved
(Phase H's sealed evaluation is complete; all four gates passed). Holdout
labels are incorporated HERE ONLY as resolved historical state for this
forward-looking system -- they are NEVER used to fit a model, fit a scaler,
select a feature, select a threshold, or make any further evaluation/tuning
decision. `build_initial_state` does not fit or evaluate anything; it only
aggregates already-resolved historical labels.

**Phase F update policy: RESOLUTION-ONLY.** A transaction is NEVER added to
`phase_f_counts`/`phase_f_global_*` merely because it occurred -- only
`update_resolved_labels`, called explicitly with a known outcome, updates
these counters. IEEE-CIS has no real `label_resolved_at` field; this models
"resolved" as "the moment `update_resolved_labels` is invoked," not any true
confirmation timestamp -- documented here as an explicit, honest
simplification, not a claim of real-world fidelity.

Counting (`update_resolved_labels`) is a plain commutative aggregation --
unlike Phase D/F's own per-row CAUSAL computations (which care about
strict ordering and same-timestamp ties), summing resolved-label counts
does not depend on the order or grouping in which resolutions are applied.

**Phase D update policy: OCCURRENCE-ONLY, via `record_observed_transactions`.**
Unlike Phase F, Phase D never needs a label at all -- a transaction's mere
occurrence (its `payment_proxy_key`, `TransactionDT`, `TransactionAmt`) is
exactly what `phase_d_buffer` already stores, so recording it is a plain
append, never a resolution. `record_observed_transactions` never reads or
requires `isFraud` and never touches `phase_f_counts`/`phase_f_global_*` --
occurrence and label-resolution are deliberately separate paths through
separate functions (this one and `update_resolved_labels`, respectively),
each responsible for exactly one phase's state. The CALL-ORDER CONTRACT a
caller must honor is: score a transaction first (against the state as it
stood BEFORE recording), THEN record it -- never the reverse, or a
transaction could see its own occurrence in its own score. This function
cannot enforce that ordering itself (it never computes a score), but
Phase D's own bucket-tie rule (`prior_group_windowed_robust_stats` in
`sentinelpay.data.history` -- rows sharing a `(group, TransactionDT)`
bucket never see each other, whole buckets only) makes same-timestamp
recording safe regardless of order: two transactions recorded with the
SAME `TransactionDT` (one at a time or as one batch) can never end up in
each other's prior window, because they always collapse into one bucket
that is never "strictly before itself." See
`tests/test_inference_state.py`'s adversarial same-timestamp tests.

===========================================================================
IDEMPOTENCY CONTRACT -- TransactionID-keyed de-duplication
===========================================================================

Both incremental update paths are IDEMPOTENT under replay, keyed SOLELY by
`TransactionID` -- never by `payment_proxy_key`/`TransactionDT`/
`TransactionAmt`/`isFraud` or any hash of those (two distinct transactions
can legitimately share all of those, e.g. two identical-amount purchases by
the same card in the same second; they must never be treated as the same
transaction).

**Separate persisted PAYLOAD-CARRYING registries, one per phase, NEVER
reconstructed from aggregates or from the buffer.**
`InferenceState.phase_d_processed` is a DataFrame indexed by `TransactionID`
holding every occurrence `record_observed_transactions` has ever accepted
(columns: `PAYMENT_GROUP_COL`, `TransactionDT`, `TransactionAmt` -- the same
payload `phase_d_buffer` stores, but keyed by ID for O(1) lookup, which
`phase_d_buffer` deliberately is not). `InferenceState.phase_f_processed` is
the analogous registry for `update_resolved_labels` (columns:
`PAYMENT_GROUP_COL`, `isFraud`). Both are seeded by `build_initial_state`'s
bulk historical load. These registries -- NOT `phase_f_counts` (which only
holds summed `fraud_count`/`event_count`, and cannot answer "was this ID
already resolved," let alone "with what payload") and NOT `phase_d_buffer`
(which never carries `TransactionID` at all) -- are the sole source of
truth for de-duplication. This is a deliberate choice: relying on the
buffer or the aggregate counts as the duplicate registry would make
conflict detection (below) impossible and would conflate "how many events
happened" with "which specific IDs have been seen."

**The two registries are INDEPENDENT domains.** A `TransactionID` appearing
in `phase_d_processed` has no effect on `phase_f_processed` membership
checks, and vice versa -- recording a transaction's occurrence and later
resolving its label are legitimate, expected, UNRELATED events for the same
ID, matching how a real transaction is first seen and only later confirmed
fraudulent/legitimate. Neither path blocks or is blocked by the other
sharing the same ID.

===========================================================================
CONFLICT POLICY -- exact replay vs. conflicting resubmission
===========================================================================

A `TransactionID` can be resubmitted (to the same function) either within
one batch (repeated rows) or across separate calls (including after a
save/load round-trip). Both functions apply IDENTICAL rules, checked against
the SAME payload columns used for comparison (`PAYMENT_GROUP_COL`/
`TransactionDT`/`TransactionAmt` for Phase D; `PAYMENT_GROUP_COL`/`isFraud`
for Phase F):

- **Same ID + IDENTICAL payload -> idempotent no-op.** The resubmission
  changes nothing (buffer/counts unaffected); metadata still reports it as
  a skipped duplicate.
- **Same ID + DIFFERENT payload -> raises `ValueError` immediately**, before
  any state mutation. Silently accepting the new payload (overwrite) or
  silently keeping the old one (ignore) are both judged worse than failing
  loudly here: either choice would hide a real data-integrity problem
  (a TransactionID must denote exactly one real-world transaction with
  exactly one payload; a conflict means either the caller or the upstream
  data pipeline is wrong, and this function has no basis to decide which
  version is correct). This check applies uniformly whether the conflicting
  rows are two rows of the SAME batch or a new call clashing with the
  persisted registry.

**`TransactionID` is a REQUIRED column for both functions -- a BREAKING
CONTRACT CHANGE** from this module's original design (which accepted only
`payment_proxy_key`/`TransactionDT`/`TransactionAmt` for
`record_observed_transactions`, and `payment_proxy_key`/`isFraud` for
`update_resolved_labels`). Its absence now raises `ValueError` immediately
rather than silently proceeding non-idempotently -- chosen explicitly over
a softer default because there is no compatibility constraint elsewhere in
this codebase forcing one: neither function is called by any frozen Phase
G/H code (both are new Phase I entry points with no other caller), and
`score_transaction` (the one function that DOES get called on every
transaction) never calls either of these two functions itself, so
tightening their input contract does not ripple into the scoring path at
all. `score_transaction` itself is UNCHANGED and still does not require
`TransactionID` -- it computes a probability, not identity-tracked state,
so it has nothing to de-duplicate.

===========================================================================
STATE SCHEMA -- adds `phase_d_processed`/`phase_f_processed`
===========================================================================

`save_state` writes two additional files, `phase_d_processed.parquet` and
`phase_f_processed.parquet` (each a `TransactionID` column plus that
phase's payload columns), alongside the three that already existed
(`phase_d_buffer.parquet`, `phase_f_counts.parquet`, `state_meta.json`).

**Backward compatibility / migration:** `load_state` treats either new file
being ABSENT as "no IDs tracked yet" (an empty registry), not an error -- so
a state directory saved by the pre-idempotency version of this module loads
without modification. This is a real, honest limitation, not a transparent
upgrade: a directory produced by the old code has NO record of which
`TransactionID`s (or payloads) were already recorded/resolved before the
upgrade, so replaying a PRE-upgrade transaction against a freshly-loaded
legacy directory will NOT be caught (it will double-count exactly as
before, and cannot be conflict-checked either). Idempotency and conflict
detection are only guaranteed for transactions recorded/resolved AFTER a
directory has been saved at least once by this schema version (or for a
directory built fresh via `build_initial_state`, which seeds both
registries from its own historical load). There is no in-place schema
migration that retroactively recovers historical payloads for an old
directory -- rebuilding via `build_initial_state` is the only way to get a
legacy directory onto solid idempotent footing.

===========================================================================
SCOPE LIMITATION -- registries are UNBOUNDED (Phase I / backtesting scope)
===========================================================================

`phase_d_processed` and `phase_f_processed` grow monotonically for the
life of a state directory -- there is NO TTL, size cap, pruning, or
expiry, by deliberate choice (a design review considered and rejected
adding one; see that review for the full trade-off). Every accepted
`TransactionID` is retained forever, in memory and on disk, for as long as
the directory exists. This is a genuine, permanent-until-reconsidered
limitation, not an oversight -- acceptable ONLY because Phase I's actual
scope is a single-process CLI operating over a FIXED, finite historical
dataset (IEEE-CIS, on the order of ~590K rows total), not a live,
unbounded, multi-instance production stream. If Phase I ever becomes a
genuinely long-running service, this needs a deliberate, separately
reviewed retention policy BEFORE that happens -- not a fixed retention
window invented here without review (this project's own discipline
requires any such constant be declared and justified before the function
that would use it is built). Equally out of scope for the same reason:
external dedup infrastructure (a real KV store, a database with TTL) and
concurrent-writer/locking support for `save_state`/`load_state` (currently
a single-process, whole-file-rewrite design with no concurrency story) --
both are the right answer for an actual live deployment, neither is
needed for Phase I's current backtesting/prototype shape, and neither is
implemented here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sentinelpay.config import DataConfig, load_config
from sentinelpay.data.split import PARTITION_ORDER, SplitConfig, load_split_config
from sentinelpay.eda.grouping_key_sufficiency import build_group_key
from sentinelpay.eda.run_phase_h import build_full_frame
from sentinelpay.model_features import PAYMENT_GROUP_COL

# Phase F's state source -- embargo partitions EXCLUDED, matching
# target_history.py's own frozen contract exactly.
PHASE_F_STATE_SOURCE_PARTITIONS = {"train", "validation", "holdout"}
# Phase D's state source -- ALL FIVE, unrestricted, matching detection.py's
# own already-approved non-target semantics.
PHASE_D_STATE_SOURCE_PARTITIONS = set(PARTITION_ORDER)

PHASE_D_BUFFER_COLUMNS = [PAYMENT_GROUP_COL, "TransactionDT", "TransactionAmt"]
PHASE_F_COUNT_COLUMNS = ["fraud_count", "event_count"]

TRANSACTION_ID_COL = "TransactionID"
# Payload columns compared for the CONFLICT POLICY -- see module docstring.
# Phase D's registry payload is exactly phase_d_buffer's own schema (it is
# simply indexed by TransactionID in addition); Phase F's is the minimal
# pair needed to detect a re-resolved transaction with a different label.
PHASE_D_PROCESSED_PAYLOAD_COLUMNS = PHASE_D_BUFFER_COLUMNS
PHASE_F_PROCESSED_PAYLOAD_COLUMNS = [PAYMENT_GROUP_COL, "isFraud"]


def _empty_registry(payload_columns: list[str]) -> pd.DataFrame:
    """An empty, properly-dtyped `TransactionID`-indexed registry -- used
    both as the `InferenceState` dataclass default and as `load_state`'s
    backward-compatibility fallback when a registry file is absent."""
    df = pd.DataFrame({TRANSACTION_ID_COL: pd.Series(dtype="int64")})
    for col in payload_columns:
        df[col] = pd.Series(dtype="object")
    return df.set_index(TRANSACTION_ID_COL)


@dataclass
class InferenceState:
    phase_d_buffer: pd.DataFrame  # columns: PAYMENT_GROUP_COL, TransactionDT, TransactionAmt
    phase_f_counts: pd.DataFrame  # indexed by PAYMENT_GROUP_COL; columns: fraud_count, event_count (int64)
    phase_f_global_fraud_count: int
    phase_f_global_event_count: int
    # TransactionID-indexed payload registries -- see module docstring's
    # IDEMPOTENCY CONTRACT / CONFLICT POLICY. Defaulted to empty so existing
    # direct-construction call sites (tests building an InferenceState by
    # hand) keep working; an empty registry simply means "no replay history
    # known yet," the correct conservative default for a hand-built state.
    phase_d_processed: pd.DataFrame = field(default_factory=lambda: _empty_registry(PHASE_D_PROCESSED_PAYLOAD_COLUMNS))
    phase_f_processed: pd.DataFrame = field(default_factory=lambda: _empty_registry(PHASE_F_PROCESSED_PAYLOAD_COLUMNS))
    metadata: dict = field(default_factory=dict)


def _dedupe_and_check_conflicts(
    batch: pd.DataFrame, registry: pd.DataFrame, payload_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared CONFLICT POLICY enforcement for both update paths (see module
    docstring). `batch` must already have an int64 `TransactionID` column.

    Returns `(new_rows, deduped_batch)`:
    - `deduped_batch` is `batch` collapsed to one row per `TransactionID`
      (via `keep="first"`, safe only because same-ID rows within `batch`
      were just verified identical).
    - `new_rows` is the subset of `deduped_batch` whose `TransactionID` is
      NOT already in `registry` -- the rows the caller should actually
      apply (append to a buffer / fold into counts). Already-known IDs are
      confirmed identical to the registry and therefore correctly excluded
      (an idempotent no-op).

    Raises `ValueError` for any `TransactionID` that appears more than once
    within `batch` with a non-identical payload, or that matches a
    `registry` entry with a non-identical payload -- see CONFLICT POLICY.
    """
    # Intra-batch conflict check: any TransactionID repeated in `batch`
    # must carry an IDENTICAL payload every time it appears.
    per_id_nunique = batch.groupby(TRANSACTION_ID_COL)[payload_columns].nunique(dropna=False)
    conflicting = per_id_nunique.index[(per_id_nunique > 1).any(axis=1)]
    if len(conflicting):
        raise ValueError(
            f"conflicting payload for TransactionID(s) {sorted(conflicting.tolist())} within the same batch -- "
            "same ID resubmitted with different data is rejected, not silently accepted (see module docstring "
            "CONFLICT POLICY)"
        )

    deduped_batch = batch.drop_duplicates(subset=TRANSACTION_ID_COL, keep="first")

    # Cross-call conflict check: any TransactionID already in `registry`
    # must carry an IDENTICAL payload in this submission.
    known_mask = deduped_batch[TRANSACTION_ID_COL].isin(registry.index)
    known_rows = deduped_batch[known_mask]
    if len(known_rows):
        comparison = known_rows.set_index(TRANSACTION_ID_COL)[payload_columns].join(
            registry[payload_columns], rsuffix="_existing"
        )
        mismatch = pd.Series(False, index=comparison.index)
        for col in payload_columns:
            mismatch = mismatch | (comparison[col] != comparison[f"{col}_existing"])
        if mismatch.any():
            raise ValueError(
                f"conflicting payload for already-processed TransactionID(s) {sorted(comparison.index[mismatch].tolist())} "
                "-- same ID resubmitted with different data than previously recorded is rejected, not silently "
                "accepted or overwritten (see module docstring CONFLICT POLICY)"
            )

    new_rows = deduped_batch[~known_mask]
    return new_rows, deduped_batch


def build_initial_state(config: DataConfig | None = None, split_config: SplitConfig | None = None) -> InferenceState:
    """Builds the initial inference state from `train`+`validation`+`holdout`
    (Phase F) / all five partitions (Phase D). Does not fit or evaluate
    anything -- a pure aggregation of already-resolved historical labels.
    Never reruns Phase H; `build_full_frame` is a data loader only.
    """
    config = config or load_config()
    split_config = split_config or load_split_config()

    full, n_rows_total = build_full_frame(config, split_config)
    valid = build_group_key(full, config.payment_proxy_key_columns, key_name=PAYMENT_GROUP_COL)
    valid = valid.astype({TRANSACTION_ID_COL: "int64"})

    phase_d_buffer = valid[PHASE_D_BUFFER_COLUMNS].reset_index(drop=True)
    # Seed the Phase D registry from the SAME pool phase_d_buffer itself was
    # drawn from -- so a later replay of a HISTORICAL TransactionID via
    # record_observed_transactions is correctly recognized (and
    # conflict-checked), not silently re-appended.
    phase_d_processed = valid.set_index(TRANSACTION_ID_COL)[PHASE_D_PROCESSED_PAYLOAD_COLUMNS]

    phase_f_pool = valid[valid["partition"].isin(sorted(PHASE_F_STATE_SOURCE_PARTITIONS))]
    per_key = phase_f_pool.groupby(PAYMENT_GROUP_COL)["isFraud"].agg(fraud_count="sum", event_count="count").astype("int64")
    global_fraud_count = int(phase_f_pool["isFraud"].sum())
    global_event_count = int(len(phase_f_pool))
    # Seed the Phase F registry from the SAME pool phase_f_counts itself was
    # drawn from (train+validation+holdout only) -- matches Phase F's own
    # frozen contract exactly, same reasoning as phase_d_processed.
    phase_f_processed = phase_f_pool.set_index(TRANSACTION_ID_COL)[PHASE_F_PROCESSED_PAYLOAD_COLUMNS]

    metadata = {
        "model_fit_source": "train partition ONLY (Phase G/H regime) -- see sentinelpay.inference.artifacts",
        "phase_d_state_source": "all five partitions (train, embargo_1, validation, embargo_2, holdout) -- "
        "unrestricted, matches Phase D's own non-target semantics",
        "phase_f_state_source": "train + validation + holdout ONLY -- matches Phase F's frozen target-history contract",
        "phase_f_embargo_partitions_excluded": sorted(["embargo_1", "embargo_2"]),
        "phase_d_partitions_included": sorted(PHASE_D_STATE_SOURCE_PARTITIONS),
        "phase_f_update_policy": "resolution-only -- a transaction contributes to Phase F counters ONLY via "
        "update_resolved_labels, never at occurrence. IEEE-CIS has no real label_resolved_at field; this "
        "models resolution as happening at the moment update-resolved is invoked, not any true confirmation time.",
        "holdout_usage_note": "holdout labels are incorporated here ONLY as resolved historical state for this "
        "forward-looking inference system. Phase H's sealed evaluation (reports/eda/phase_h_results.json) is "
        "complete and was NOT rerun; this state was NOT used to select a model, scaler, feature, or threshold.",
        "n_rows_total_loaded": n_rows_total,
        "n_rows_phase_d_buffer": int(len(phase_d_buffer)),
        "n_keys_phase_f": int(len(per_key)),
        "n_ids_phase_d_observed": int(len(phase_d_processed)),
        "n_ids_phase_f_resolved": int(len(phase_f_processed)),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return InferenceState(
        phase_d_buffer=phase_d_buffer,
        phase_f_counts=per_key[PHASE_F_COUNT_COLUMNS],
        phase_f_global_fraud_count=global_fraud_count,
        phase_f_global_event_count=global_event_count,
        phase_d_processed=phase_d_processed,
        phase_f_processed=phase_f_processed,
        metadata=metadata,
    )


def _save_registry(registry: pd.DataFrame, path: Path) -> None:
    registry.reset_index().to_parquet(path, index=False)


def _load_registry(path: Path, payload_columns: list[str]) -> pd.DataFrame:
    # Absent file = pre-idempotency state directory -- see module
    # docstring's STATE SCHEMA / backward-compatibility note. Not an error:
    # treated as "no IDs tracked yet."
    if not path.exists():
        return _empty_registry(payload_columns)
    df = pd.read_parquet(path).astype({TRANSACTION_ID_COL: "int64"}).set_index(TRANSACTION_ID_COL)
    return df[payload_columns]


def save_state(state: InferenceState, dir_path: Path) -> None:
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    state.phase_d_buffer.to_parquet(dir_path / "phase_d_buffer.parquet", index=False)
    state.phase_f_counts.reset_index().rename(columns={"index": PAYMENT_GROUP_COL}).to_parquet(
        dir_path / "phase_f_counts.parquet", index=False
    )
    _save_registry(state.phase_d_processed, dir_path / "phase_d_processed.parquet")
    _save_registry(state.phase_f_processed, dir_path / "phase_f_processed.parquet")
    meta = dict(state.metadata)
    meta["phase_f_global_fraud_count"] = state.phase_f_global_fraud_count
    meta["phase_f_global_event_count"] = state.phase_f_global_event_count
    (dir_path / "state_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_state(dir_path: Path) -> InferenceState:
    dir_path = Path(dir_path)
    phase_d_buffer = pd.read_parquet(dir_path / "phase_d_buffer.parquet")
    phase_f_counts = pd.read_parquet(dir_path / "phase_f_counts.parquet").set_index(PAYMENT_GROUP_COL)
    phase_d_processed = _load_registry(dir_path / "phase_d_processed.parquet", PHASE_D_PROCESSED_PAYLOAD_COLUMNS)
    phase_f_processed = _load_registry(dir_path / "phase_f_processed.parquet", PHASE_F_PROCESSED_PAYLOAD_COLUMNS)
    meta = json.loads((dir_path / "state_meta.json").read_text(encoding="utf-8"))
    return InferenceState(
        phase_d_buffer=phase_d_buffer,
        phase_f_counts=phase_f_counts[PHASE_F_COUNT_COLUMNS].astype("int64"),
        phase_f_global_fraud_count=int(meta["phase_f_global_fraud_count"]),
        phase_f_global_event_count=int(meta["phase_f_global_event_count"]),
        phase_d_processed=phase_d_processed,
        phase_f_processed=phase_f_processed,
        metadata=meta,
    )


def update_resolved_labels(state: InferenceState, resolved: pd.DataFrame, config: DataConfig | None = None) -> InferenceState:
    """RESOLUTION-ONLY, IDEMPOTENT Phase F update (see module docstring's
    IDEMPOTENCY CONTRACT / CONFLICT POLICY). `resolved` must have `isFraud`
    (0/1) and `TransactionID` columns, and either `PAYMENT_GROUP_COL`
    directly or the raw `payment_proxy_key` component columns (built via
    `build_group_key`, unmodified).

    A `TransactionID` already present in `state.phase_f_processed` (from a
    prior call, from a same-batch duplicate, or from
    `build_initial_state`'s historical seed) with an IDENTICAL
    `(PAYMENT_GROUP_COL, isFraud)` payload is skipped -- its `isFraud` is
    NOT re-added to `phase_f_counts`/`phase_f_global_*` (idempotent no-op).
    The SAME `TransactionID` resubmitted with a DIFFERENT payload raises
    `ValueError` instead -- see CONFLICT POLICY.

    Returns a NEW `InferenceState` -- `state`'s own DataFrames are never
    mutated in place. Phase D's buffer and registry are untouched by this
    function (Phase D never needs a label, and the two registries are
    independent -- see IDEMPOTENCY CONTRACT)."""
    if "isFraud" not in resolved.columns:
        raise ValueError("update_resolved_labels requires an 'isFraud' column")
    if TRANSACTION_ID_COL not in resolved.columns:
        raise ValueError(
            "update_resolved_labels requires a 'TransactionID' column for resolution de-duplication -- see "
            "module docstring IDEMPOTENCY CONTRACT. A transaction's real ID must be supplied, not omitted."
        )

    if PAYMENT_GROUP_COL not in resolved.columns:
        config = config or load_config()
        resolved = build_group_key(resolved, config.payment_proxy_key_columns, key_name=PAYMENT_GROUP_COL)

    resolved = resolved.astype({TRANSACTION_ID_COL: "int64", "isFraud": "int64"})
    new_rows, deduped_batch = _dedupe_and_check_conflicts(resolved, state.phase_f_processed, PHASE_F_PROCESSED_PAYLOAD_COLUMNS)

    delta = new_rows.groupby(PAYMENT_GROUP_COL)["isFraud"].agg(fraud_count="sum", event_count="count").astype("int64")
    new_counts = state.phase_f_counts.add(delta, fill_value=0).astype("int64")

    new_global_fraud = state.phase_f_global_fraud_count + int(new_rows["isFraud"].sum())
    new_global_event = state.phase_f_global_event_count + int(len(new_rows))

    new_registry_rows = new_rows.set_index(TRANSACTION_ID_COL)[PHASE_F_PROCESSED_PAYLOAD_COLUMNS]
    new_processed = (
        pd.concat([state.phase_f_processed, new_registry_rows]) if len(state.phase_f_processed) else new_registry_rows
    )

    new_metadata = dict(state.metadata)
    new_metadata["last_resolved_update_utc"] = datetime.now(timezone.utc).isoformat()
    new_metadata["n_keys_phase_f"] = int(len(new_counts))
    new_metadata["n_ids_phase_f_resolved"] = int(len(new_processed))
    new_metadata["n_duplicate_ids_skipped_last_resolved_update"] = int(len(resolved) - len(new_rows))

    return InferenceState(
        phase_d_buffer=state.phase_d_buffer,
        phase_f_counts=new_counts,
        phase_f_global_fraud_count=new_global_fraud,
        phase_f_global_event_count=new_global_event,
        phase_d_processed=state.phase_d_processed,
        phase_f_processed=new_processed,
        metadata=new_metadata,
    )


def record_observed_transactions(
    state: InferenceState, observed: pd.DataFrame, config: DataConfig | None = None
) -> InferenceState:
    """OCCURRENCE-ONLY, IDEMPOTENT Phase D update (see module docstring's
    IDEMPOTENCY CONTRACT / CONFLICT POLICY). `observed` must have
    `TransactionDT`/`TransactionAmt`/`TransactionID` columns and either
    `PAYMENT_GROUP_COL` directly or the raw `payment_proxy_key` component
    columns (built via `build_group_key`, unmodified).

    A `TransactionID` already present in `state.phase_d_processed` (from a
    prior call, from a same-batch duplicate, or from
    `build_initial_state`'s historical seed) with an IDENTICAL
    `(PAYMENT_GROUP_COL, TransactionDT, TransactionAmt)` payload is skipped
    -- it is NOT appended to `phase_d_buffer` again (idempotent no-op). The
    SAME `TransactionID` resubmitted with a DIFFERENT payload raises
    `ValueError` instead -- see CONFLICT POLICY. An `isFraud` column, if
    present in `observed`, is simply never selected or referenced -- this
    function works identically with or without one.

    Returns a NEW `InferenceState` -- `state`'s own DataFrames are never
    mutated in place. Phase F's counters and registry are untouched by this
    function; only `update_resolved_labels` grows those (see module
    docstring for the CALL-ORDER CONTRACT and the bucket-tie argument for
    why same-`TransactionDT` batch recording is safe)."""
    if PAYMENT_GROUP_COL not in observed.columns:
        config = config or load_config()
        observed = build_group_key(observed, config.payment_proxy_key_columns, key_name=PAYMENT_GROUP_COL)

    required_columns = set(PHASE_D_BUFFER_COLUMNS) | {TRANSACTION_ID_COL}
    missing = required_columns - set(observed.columns)
    if missing:
        raise ValueError(
            f"record_observed_transactions requires columns {sorted(missing)} -- 'TransactionID' is required for "
            "occurrence de-duplication, see module docstring IDEMPOTENCY CONTRACT"
        )

    observed = observed.astype({TRANSACTION_ID_COL: "int64"})
    new_rows, deduped_batch = _dedupe_and_check_conflicts(observed, state.phase_d_processed, PHASE_D_PROCESSED_PAYLOAD_COLUMNS)

    new_buffer_rows = new_rows[PHASE_D_BUFFER_COLUMNS].reset_index(drop=True)
    new_buffer = (
        pd.concat([state.phase_d_buffer, new_buffer_rows], ignore_index=True)
        if len(state.phase_d_buffer)
        else new_buffer_rows
    )
    new_registry_rows = new_rows.set_index(TRANSACTION_ID_COL)[PHASE_D_PROCESSED_PAYLOAD_COLUMNS]
    new_processed = (
        pd.concat([state.phase_d_processed, new_registry_rows]) if len(state.phase_d_processed) else new_registry_rows
    )

    new_metadata = dict(state.metadata)
    new_metadata["last_observed_update_utc"] = datetime.now(timezone.utc).isoformat()
    new_metadata["n_rows_phase_d_buffer"] = int(len(new_buffer))
    new_metadata["n_ids_phase_d_observed"] = int(len(new_processed))
    new_metadata["n_duplicate_ids_skipped_last_observed_update"] = int(len(observed) - len(new_rows))

    return InferenceState(
        phase_d_buffer=new_buffer,
        phase_f_counts=state.phase_f_counts,
        phase_f_global_fraud_count=state.phase_f_global_fraud_count,
        phase_f_global_event_count=state.phase_f_global_event_count,
        phase_d_processed=new_processed,
        phase_f_processed=state.phase_f_processed,
        metadata=new_metadata,
    )

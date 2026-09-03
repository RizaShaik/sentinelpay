"""Phase I minimal CLI: build artifacts/state, score a transaction,
record a scored transaction's occurrence into Phase D state, and/or update
resolved Phase F label state. No threshold, no fraud/legitimate decision --
every command surfaces raw probabilities/diagnostics only.

Run with:
    .venv\\Scripts\\python.exe -m sentinelpay.inference.cli build
    .venv\\Scripts\\python.exe -m sentinelpay.inference.cli score --input txn.json
    .venv\\Scripts\\python.exe -m sentinelpay.inference.cli record-observed --input txn.json
    .venv\\Scripts\\python.exe -m sentinelpay.inference.cli update-resolved --input resolved.json

`record-observed` must be called AFTER `score` for the same transaction
(against the state as it stood before recording) -- see
`sentinelpay.inference.state.record_observed_transactions`'s CALL-ORDER
CONTRACT. It grows Phase D state only; `update-resolved` grows Phase F
state only -- the two never overlap.

Both `record-observed` and `update-resolved` require a `TransactionID`
field in their input (a breaking contract requirement -- see
`sentinelpay.inference.state`'s IDEMPOTENCY CONTRACT): each is idempotent
under replay, keyed solely by `TransactionID`, so re-running either command
with the same transaction is always safe and a no-op the second time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from sentinelpay.config import load_detection_config
from sentinelpay.inference.artifacts import build_f2_artifact, load_artifact, save_artifact
from sentinelpay.inference.scoring import score_transaction
from sentinelpay.inference.state import (
    build_initial_state,
    load_state,
    record_observed_transactions,
    save_state,
    update_resolved_labels,
)

DEFAULT_ARTIFACT_PATH = Path("artifacts/inference/model.joblib")
DEFAULT_STATE_DIR = Path("artifacts/inference/state")


def cmd_build(args: argparse.Namespace) -> None:
    print("Building F2 model artifact (train partition ONLY -- Phase G/H regime)...", file=sys.stderr)
    artifact = build_f2_artifact()
    save_artifact(artifact, Path(args.artifact_path))
    print(f"Saved model artifact: {args.artifact_path}", file=sys.stderr)

    print(
        "Building inference state (Phase D: all five partitions; Phase F: train+validation+holdout, "
        "embargo_1/embargo_2 excluded)...",
        file=sys.stderr,
    )
    state = build_initial_state()
    save_state(state, Path(args.state_dir))
    print(f"Saved inference state: {args.state_dir}", file=sys.stderr)
    print(json.dumps({"model_metadata": artifact["metadata"], "state_metadata": state.metadata}, indent=2, default=str))


def cmd_score(args: argparse.Namespace) -> None:
    artifact = load_artifact(Path(args.artifact_path))
    state = load_state(Path(args.state_dir))
    detection_config = load_detection_config()
    transaction = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = score_transaction(transaction, state, artifact, detection_config)
    print(
        json.dumps(
            {
                "payment_proxy_key": result.payment_proxy_key,
                "fraud_probability": result.fraud_probability,
                "phase_d_diagnostics": result.phase_d_diagnostics,
                "phase_f_diagnostics": result.phase_f_diagnostics,
                "features": result.features,
            },
            indent=2,
        )
    )


def cmd_record_observed(args: argparse.Namespace) -> None:
    state = load_state(Path(args.state_dir))
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    observed_df = pd.DataFrame(records if isinstance(records, list) else [records])
    new_state = record_observed_transactions(state, observed_df)
    save_state(new_state, Path(args.state_dir))
    n_input = len(observed_df)
    n_duplicates = new_state.metadata["n_duplicate_ids_skipped_last_observed_update"]
    n_new = n_input - n_duplicates
    print(
        f"Processed {n_input} observed transaction(s) for Phase D state: {n_new} newly recorded, "
        f"{n_duplicates} duplicate(s) skipped ({args.state_dir})",
        file=sys.stderr,
    )


def cmd_update_resolved(args: argparse.Namespace) -> None:
    state = load_state(Path(args.state_dir))
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    resolved_df = pd.DataFrame(records)
    new_state = update_resolved_labels(state, resolved_df)
    save_state(new_state, Path(args.state_dir))
    n_input = len(resolved_df)
    n_duplicates = new_state.metadata["n_duplicate_ids_skipped_last_resolved_update"]
    n_new = n_input - n_duplicates
    print(
        f"Processed {n_input} resolved label(s) for Phase F state: {n_new} newly updated, "
        f"{n_duplicates} duplicate(s) skipped ({args.state_dir})",
        file=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinelpay.inference.cli")
    parser.add_argument("--artifact-path", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build")
    p_score = sub.add_parser("score")
    p_score.add_argument("--input", required=True)
    p_record = sub.add_parser("record-observed")
    p_record.add_argument("--input", required=True)
    p_update = sub.add_parser("update-resolved")
    p_update.add_argument("--input", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        cmd_build(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "record-observed":
        cmd_record_observed(args)
    elif args.command == "update-resolved":
        cmd_update_resolved(args)


if __name__ == "__main__":
    main()

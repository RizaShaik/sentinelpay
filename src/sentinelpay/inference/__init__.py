"""Phase I: the reusable inference/application layer for SentinelPay's
frozen, sealed-holdout-confirmed F2 design (Phases F/G/H). Everything under
this package is NEW code -- `sentinelpay.target_history`,
`sentinelpay.model_features`, `sentinelpay.model_evaluation`,
`sentinelpay.eda.run_phase_g`, and `sentinelpay.eda.run_phase_h` are all
imported from, never modified.

See `sentinelpay.inference.artifacts` for the frozen model artifact,
`sentinelpay.inference.state` for the per-key/global inference state, and
`sentinelpay.inference.scoring` for state-backed transaction scoring.
"""

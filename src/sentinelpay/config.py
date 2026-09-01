"""Project configuration loading.

Single source of truth for raw data paths, the small set of column-typing
facts that are safe to hardcode, the correlation-analysis cost mode, and the
candidate entity proxy-key column sets (see configs/data.yaml for provenance
notes on all of these).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
DEFAULT_DETECTION_CONFIG_PATH = PROJECT_ROOT / "configs" / "detection.yaml"


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    files: dict[str, str]
    join_key: str
    dt_column: str
    seconds_per_day: int
    total_span_days: int
    known_categorical_transaction: list[str]
    correlation_mode: str
    correlation_curated_columns: list[str]
    payment_proxy_key_columns: list[str]
    device_proxy_key_columns: list[str]

    def raw_path(self, name: str) -> Path:
        return self.raw_dir / self.files[name]


def _resolve(root: Path, rel: str) -> Path:
    return (root / rel).resolve()


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> DataConfig:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    root = PROJECT_ROOT
    paths = raw["paths"]
    correlation = raw.get("correlation", {"mode": "curated", "curated_columns": []})
    proxy_keys = raw.get("proxy_keys", {"payment_proxy_key": [], "device_proxy_key": []})

    mode = correlation.get("mode", "curated")
    if mode not in ("curated", "full"):
        raise ValueError(f"configs/data.yaml correlation.mode must be 'curated' or 'full', got {mode!r}")

    return DataConfig(
        raw_dir=_resolve(root, paths["raw_dir"]),
        interim_dir=_resolve(root, paths["interim_dir"]),
        processed_dir=_resolve(root, paths["processed_dir"]),
        reports_dir=_resolve(root, paths["reports_dir"]),
        files=raw["files"],
        join_key=raw["join"]["key"],
        dt_column=raw["temporal"]["dt_column"],
        seconds_per_day=raw["temporal"]["seconds_per_day"],
        total_span_days=raw["temporal"]["total_span_days"],
        known_categorical_transaction=raw["known_categorical"]["transaction"],
        correlation_mode=mode,
        correlation_curated_columns=correlation.get("curated_columns", []),
        payment_proxy_key_columns=proxy_keys.get("payment_proxy_key", []),
        device_proxy_key_columns=proxy_keys.get("device_proxy_key", []),
    )


@dataclass(frozen=True)
class DetectionConfig:
    """Fixed, pre-declared hyperparameters for sentinelpay.detection. Every
    fixed detector parameter lives in configs/detection.yaml -- see that
    file for justification of each value. None may be selected, tuned, or
    changed based on validation isFraud results."""

    min_history_for_score: int
    window_size_events: int
    modified_zscore_scale_constant: float
    modified_zscore_threshold: float
    zero_mad_epsilon: float


def load_detection_config(path: Path | str = DEFAULT_DETECTION_CONFIG_PATH) -> DetectionConfig:
    """Parse configs/detection.yaml. Raises ValueError on a malformed config
    (missing/non-numeric field) -- a schema check, not a data-driven check;
    this module never looks at transaction data or isFraud."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    try:
        return DetectionConfig(
            min_history_for_score=int(raw["min_history_for_score"]),
            window_size_events=int(raw["window_size_events"]),
            modified_zscore_scale_constant=float(raw["modified_zscore_scale_constant"]),
            modified_zscore_threshold=float(raw["modified_zscore_threshold"]),
            zero_mad_epsilon=float(raw["zero_mad_epsilon"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("configs/detection.yaml is missing or has invalid required field(s)") from exc

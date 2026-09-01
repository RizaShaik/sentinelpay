import pytest
import yaml

from sentinelpay.config import DetectionConfig, load_detection_config


def _write_yaml(tmp_path, data):
    path = tmp_path / "detection.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return path


def test_load_detection_config_parses_valid_yaml(tmp_path):
    path = _write_yaml(
        tmp_path,
        {
            "min_history_for_score": 5,
            "window_size_events": 20,
            "modified_zscore_scale_constant": 0.6745,
            "modified_zscore_threshold": 3.5,
            "zero_mad_epsilon": 1e-9,
        },
    )
    cfg = load_detection_config(path)

    assert cfg == DetectionConfig(
        min_history_for_score=5,
        window_size_events=20,
        modified_zscore_scale_constant=0.6745,
        modified_zscore_threshold=3.5,
        zero_mad_epsilon=1e-9,
    )
    # Numeric fields are coerced to the declared type regardless of the YAML's own type.
    assert isinstance(cfg.min_history_for_score, int)
    assert isinstance(cfg.window_size_events, int)
    assert isinstance(cfg.modified_zscore_scale_constant, float)
    assert isinstance(cfg.modified_zscore_threshold, float)
    assert isinstance(cfg.zero_mad_epsilon, float)


def test_load_detection_config_coerces_numeric_strings(tmp_path):
    # YAML scalars parsed as strings should still coerce cleanly via int()/float().
    path = _write_yaml(
        tmp_path,
        {
            "min_history_for_score": "5",
            "window_size_events": "20",
            "modified_zscore_scale_constant": "0.6745",
            "modified_zscore_threshold": "3.5",
            "zero_mad_epsilon": "1e-9",
        },
    )
    cfg = load_detection_config(path)
    assert cfg.min_history_for_score == 5
    assert cfg.window_size_events == 20


@pytest.mark.parametrize(
    "missing_field",
    [
        "min_history_for_score",
        "window_size_events",
        "modified_zscore_scale_constant",
        "modified_zscore_threshold",
        "zero_mad_epsilon",
    ],
)
def test_load_detection_config_raises_on_missing_required_field(tmp_path, missing_field):
    data = {
        "min_history_for_score": 5,
        "window_size_events": 20,
        "modified_zscore_scale_constant": 0.6745,
        "modified_zscore_threshold": 3.5,
        "zero_mad_epsilon": 1e-9,
    }
    del data[missing_field]
    path = _write_yaml(tmp_path, data)

    with pytest.raises(ValueError):
        load_detection_config(path)


def test_load_detection_config_raises_on_non_numeric_field(tmp_path):
    path = _write_yaml(
        tmp_path,
        {
            "min_history_for_score": "not_a_number",
            "window_size_events": 20,
            "modified_zscore_scale_constant": 0.6745,
            "modified_zscore_threshold": 3.5,
            "zero_mad_epsilon": 1e-9,
        },
    )
    with pytest.raises(ValueError):
        load_detection_config(path)

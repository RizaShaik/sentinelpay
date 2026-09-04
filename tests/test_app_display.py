"""Regression coverage for the Investigate a Payment page's dollar-scale
display fix (app.py's fmt_amt_from_log1p / stat_row usage in render_score_result).

`sentinelpay.detection.compute_behavioral_change_score` computes
`prior_median`/`prior_mad` from `prior_group_windowed_robust_stats` over
`amt_log1p = np.log1p(TransactionAmt)`, NOT over raw dollar amounts (see
sentinelpay/data/history.py:204-206 and sentinelpay/detection.py:84-101).
These tests pin down, against the real unmodified backend function (not a
reimplementation), the exact mathematical property app.py's display fix
relies on:

- `prior_median` is safe to convert to a real dollar figure via
  `math.expm1`, because the median commutes with any strictly monotonic
  transform: median(log1p(x)) == log1p(median(x)).
- `prior_mad` (a spread statistic) does NOT commute with a nonlinear
  transform -- expm1(prior_mad) is NOT the dollar-scale MAD, so no such
  blind conversion is valid. This is demonstrated by counterexample rather
  than asserted by reasoning alone.
"""
import math
import statistics

import pandas as pd
import pytest

from sentinelpay.data.history import prior_group_windowed_robust_stats


def _score_new_row(prior_amounts: list[float], new_amount: float, window_size_events: int):
    """Builds a tiny synthetic (group, dt, amount) frame -- one row per prior
    amount plus one new row strictly after all of them -- and returns the new
    row's (prior_median, prior_mad) in amt_log1p space, exactly as
    sentinelpay.detection.compute_behavioral_change_score derives them."""
    n = len(prior_amounts)
    df = pd.DataFrame(
        {
            "synthetic_group": ["G"] * (n + 1),
            "TransactionDT": list(range(100, 100 + n)) + [1000],
            "amt_log1p": [math.log1p(a) for a in prior_amounts] + [math.log1p(new_amount)],
        }
    )
    out = prior_group_windowed_robust_stats(
        df, group_col="synthetic_group", amount_col="amt_log1p", dt_col="TransactionDT",
        window_size_events=window_size_events,
    )
    new_row = out.iloc[-1]
    assert int(new_row["prior_count_in_window"]) == n
    return float(new_row["prior_median"]), float(new_row["prior_mad"])


@pytest.mark.parametrize(
    "prior_amounts",
    [
        [8.0, 9.0, 10.0, 11.0, 12.0],
        [5.0, 50.0, 500.0, 5000.0, 50000.0],
        [19.99, 19.99, 19.99, 250.0, 4.5, 19.99, 19.99],
    ],
)
def test_expm1_of_prior_median_recovers_true_dollar_median(prior_amounts):
    prior_median_log1p, _prior_mad_log1p = _score_new_row(prior_amounts, new_amount=10.0, window_size_events=20)
    dollar_median = math.expm1(prior_median_log1p)
    assert dollar_median == pytest.approx(statistics.median(prior_amounts), rel=1e-9)


@pytest.mark.parametrize(
    "prior_amounts",
    [
        [8.0, 9.0, 10.0, 11.0, 12.0],
        [5.0, 50.0, 500.0, 5000.0, 50000.0],
    ],
)
def test_expm1_of_prior_mad_is_not_the_dollar_scale_mad(prior_amounts):
    """Counterexample proving no blind expm1(prior_mad) conversion is valid:
    MAD does not commute with the nonlinear log1p transform the way the
    median does."""
    prior_median_log1p, prior_mad_log1p = _score_new_row(prior_amounts, new_amount=10.0, window_size_events=20)
    dollar_median = statistics.median(prior_amounts)
    dollar_mad = statistics.median(abs(a - dollar_median) for a in prior_amounts)

    naive_conversion = math.expm1(prior_mad_log1p)
    assert naive_conversion != pytest.approx(dollar_mad, rel=1e-2)


def test_prior_mad_is_zero_when_all_prior_amounts_are_identical():
    # Sanity check the fixture/helper itself: identical prior amounts must
    # give MAD == 0 exactly (median-of-zeros), not merely "small".
    _prior_median, prior_mad = _score_new_row([10.0, 10.0, 10.0], new_amount=10.0, window_size_events=20)
    assert prior_mad == 0.0

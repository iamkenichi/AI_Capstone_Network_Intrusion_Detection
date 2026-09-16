"""
Tests for the evaluation helpers whose output appears directly in the reports.

The Wilson interval is tested because the bias analysis uses it to qualify
per-family recall on samples as small as 43 flows, and a wrong interval there
would make an over-confident claim look rigorous.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import evaluate


# --------------------------------------------------------------------------- #
# Wilson score interval
# --------------------------------------------------------------------------- #
def test_wilson_interval_brackets_the_point_estimate() -> None:
    for successes, trials in [(1, 10), (5, 10), (9, 10), (99, 167), (12430, 13642)]:
        low, high = evaluate.wilson_interval(successes, trials)
        phat = successes / trials
        assert low <= phat <= high
        assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0


def test_wilson_interval_stays_inside_the_unit_range_at_the_boundaries() -> None:
    """The reason Wilson is used instead of the normal approximation.

    At 43/43 the normal approximation gives a zero-width interval at 1.0, which
    would claim certainty from 43 observations. Wilson does not.
    """
    low, high = evaluate.wilson_interval(43, 43)
    assert high == 1.0
    assert 0.85 < low < 1.0, f"expected a non-degenerate lower bound, got {low}"

    low_zero, high_zero = evaluate.wilson_interval(0, 30)
    assert low_zero == 0.0
    assert 0.0 < high_zero < 0.20


def test_wilson_interval_narrows_as_the_sample_grows() -> None:
    widths = []
    for trials in (20, 200, 2000, 20000):
        low, high = evaluate.wilson_interval(round(0.9 * trials), trials)
        widths.append(high - low)
    assert widths == sorted(widths, reverse=True), (
        f"interval must shrink with sample size, got {widths}")


def test_wilson_interval_handles_an_empty_group() -> None:
    low, high = evaluate.wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)


@pytest.mark.parametrize("successes,trials", [(1, 10), (5, 10), (9, 10), (43, 43), (99, 167)])
def test_wilson_interval_matches_the_score_test_it_inverts(
    successes: int, trials: int,
) -> None:
    """Check the closed form against the definition it is derived from.

    The Wilson interval is *defined* as the set of p for which the score test
    does not reject: |p_hat - p| / sqrt(p(1-p)/n) <= z. Solving that numerically
    is an independent derivation - it shares no algebra with the implementation -
    so it catches a mis-transcribed closed form, which asserting hand-computed
    constants would not.
    """
    z = 1.96
    phat = successes / trials
    low, high = evaluate.wilson_interval(successes, trials)

    def score(p: float) -> float:
        return abs(phat - p) / np.sqrt(p * (1 - p) / trials) - z

    # Just inside each bound the score test must accept; just outside, reject.
    eps = 1e-6
    for bound in (low, high):
        if 0.0 < bound < 1.0:
            inside = bound + eps if bound == low else bound - eps
            outside = bound - eps if bound == low else bound + eps
            assert score(inside) < 0, f"p={inside} should be inside the interval"
            if 0.0 < outside < 1.0:
                assert score(outside) > 0, f"p={outside} should be outside the interval"

"""Tests for standardised normal quantiles and tail expectations.

Four groups, in increasing order of what they can catch:

1. Published values — the function agrees with a statistical table.
2. Closed form vs numerical integration — the algebra is actually right.
3. Invariants — relationships that must hold for every input, not just the ones
   listed here.
4. Rejected input — bad arguments fail loudly rather than returning a plausible
   wrong number.

Group 2 is the one that matters. Groups 1 and 3 confirm the function behaves
sensibly; group 2 computes the same quantity a completely different way and
checks they agree, which is what verifies the derivation rather than the typing.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, stats

from tplab.stats.distributions import (
    standard_normal_expected_shortfall,
    standard_normal_quantile,
)

# ---------------------------------------------------------------------------
# 1. Published values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alpha, expected",
    [
        (0.10, -1.281552),
        (0.05, -1.644854),
        (0.01, -2.326348),
        (0.001, -3.090232),
    ],
)
def test_quantile_matches_published_values(alpha: float, expected: float) -> None:
    """Standard table values. -1.645 at 5% and -2.326 at 1% are worth memorising."""
    assert standard_normal_quantile(alpha) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "alpha, expected",
    [
        (0.10, -1.754983),
        (0.05, -2.062713),
        (0.01, -2.665214),
        (0.001, -3.367090),
    ],
)
def test_expected_shortfall_matches_published_values(alpha: float, expected: float) -> None:
    assert standard_normal_expected_shortfall(alpha) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Closed form against numerical integration
# ---------------------------------------------------------------------------


def _expected_shortfall_by_integration(alpha: float) -> float:
    """ES computed the slow, obvious way: integrate x*phi(x) over the tail.

    Shares no algebra with the implementation, so agreement is real evidence
    rather than the same mistake made twice.
    """
    x_alpha = stats.norm.ppf(alpha)
    integral, _abserr = integrate.quad(lambda x: x * stats.norm.pdf(x), -np.inf, x_alpha)
    return float(integral / alpha)


@pytest.mark.parametrize("alpha", [0.20, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001, 0.0001])
def test_closed_form_matches_numerical_integration(alpha: float) -> None:
    """Verifies ES = -phi(x_alpha) / alpha, the step that actually needed deriving."""
    assert standard_normal_expected_shortfall(alpha) == pytest.approx(
        _expected_shortfall_by_integration(alpha), rel=1e-9
    )


# ---------------------------------------------------------------------------
# 3. Invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.20, 0.10, 0.05, 0.01, 0.001])
def test_shortfall_is_always_worse_than_the_quantile(alpha: float) -> None:
    """ES averages the tail beyond VaR, so it must lie strictly further out."""
    assert standard_normal_expected_shortfall(alpha) < standard_normal_quantile(alpha)


@pytest.mark.parametrize("alpha", [0.20, 0.10, 0.05, 0.01, 0.001])
def test_both_measures_are_negative_in_the_loss_tail(alpha: float) -> None:
    """The sign convention, pinned. A positive value here means someone flipped it."""
    assert standard_normal_quantile(alpha) < 0
    assert standard_normal_expected_shortfall(alpha) < 0


def test_quantiles_deepen_as_alpha_shrinks() -> None:
    """A rarer event is a worse one. Monotone, strictly."""
    alphas = (0.20, 0.10, 0.05, 0.01, 0.001)
    quantiles = [standard_normal_quantile(a) for a in alphas]
    assert quantiles == sorted(quantiles, reverse=True)
    assert len(set(quantiles)) == len(quantiles)


def test_shortfalls_deepen_as_alpha_shrinks() -> None:
    alphas = (0.20, 0.10, 0.05, 0.01, 0.001)
    shortfalls = [standard_normal_expected_shortfall(a) for a in alphas]
    assert shortfalls == sorted(shortfalls, reverse=True)


def test_quantile_inverts_the_cdf() -> None:
    """Phi(Phi^-1(alpha)) == alpha. Guards against a ppf/cdf mix-up."""
    for alpha in (0.20, 0.05, 0.01, 0.001):
        assert float(stats.norm.cdf(standard_normal_quantile(alpha))) == pytest.approx(
            alpha, rel=1e-12
        )


# ---------------------------------------------------------------------------
# 4. Rejected input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, -0.01, 1.5, 99.0])
@pytest.mark.parametrize("function", [standard_normal_quantile, standard_normal_expected_shortfall])
def test_alpha_outside_the_loss_tail_is_rejected(function, alpha: float) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="tail probability"):
        function(alpha)


def test_confidence_level_mistake_is_caught_with_a_useful_message() -> None:
    """Passing 0.99 instead of 0.01 is the most likely misuse of this API.

    It is caught only because alpha is capped at 0.5. Over the full unit interval
    0.99 would be a legal request for the 99th percentile, and the caller would
    receive +2.33 where they expected a loss.
    """
    with pytest.raises(ValueError, match=r"pass alpha=0\.01, not 0\.99"):
        standard_normal_quantile(0.99)


def test_boundaries_are_exclusive() -> None:
    """Both endpoints are rejected; just inside them is accepted."""
    with pytest.raises(ValueError):
        standard_normal_quantile(0.0)
    with pytest.raises(ValueError):
        standard_normal_quantile(0.5)

    assert standard_normal_quantile(1e-12) < 0
    assert standard_normal_quantile(0.499999) < 0

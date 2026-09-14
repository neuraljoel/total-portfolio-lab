"""Tests for the parametric normal estimator and the shared interface.

The estimator itself is simple, so most of these tests are really about the
*contract* any estimator must honour: positive losses, ES at least as deep as
VaR, errors before nonsense, and linear scaling in volatility.
"""

from __future__ import annotations

import numpy as np
import pytest

from tplab.risk import MIN_OBSERVATIONS, ParametricNormal, RiskForecast, RiskModel
from tplab.stats.distributions import (
    standard_normal_expected_shortfall,
    standard_normal_quantile,
)


@pytest.fixture
def exact_sigma_returns() -> np.ndarray:
    """250 returns with sample standard deviation exactly 0.012 and mean 0.

    Half at -0.012 and half at +0.012 gives a sample variance of
    250/249 * 0.012**2, so the ddof=1 standard deviation is slightly above
    0.012. Scaling by the correction makes it exact, which lets the headline
    test assert to machine precision rather than to a tolerance.
    """
    n = 250
    raw = np.concatenate([np.full(n // 2, -0.012), np.full(n // 2, 0.012)])
    return raw * np.sqrt((n - 1) / n)


# ---------------------------------------------------------------------------
# The number itself
# ---------------------------------------------------------------------------


def test_var_is_volatility_times_the_normal_quantile(exact_sigma_returns: np.ndarray) -> None:
    """The whole model, verified against the arithmetic done by hand."""
    forecast = ParametricNormal().fit(exact_sigma_returns).forecast(alpha=0.01)

    expected = 0.012 * 2.326347874  # sigma * |Phi^-1(0.01)|
    assert forecast.value_at_risk == pytest.approx(expected, rel=1e-9)
    assert forecast.value_at_risk == pytest.approx(0.027916, abs=1e-6)


def test_es_is_volatility_times_the_normal_shortfall(exact_sigma_returns: np.ndarray) -> None:
    forecast = ParametricNormal().fit(exact_sigma_returns).forecast(alpha=0.01)

    expected = 0.012 * 2.665214220  # sigma * |phi(Phi^-1(0.01)) / 0.01|
    assert forecast.expected_shortfall == pytest.approx(expected, rel=1e-9)
    assert forecast.expected_shortfall == pytest.approx(0.031983, abs=1e-6)


def test_fitted_volatility_is_exposed(exact_sigma_returns: np.ndarray) -> None:
    model = ParametricNormal().fit(exact_sigma_returns)
    assert model.volatility == pytest.approx(0.012, rel=1e-12)
    assert model.n_observations == 250


# ---------------------------------------------------------------------------
# The contract any estimator must honour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.10, 0.05, 0.01, 0.001])
def test_losses_are_reported_as_positive_numbers(
    exact_sigma_returns: np.ndarray, alpha: float
) -> None:
    """The sign flip, pinned. Distributions are negative; this module is not."""
    forecast = ParametricNormal().fit(exact_sigma_returns).forecast(alpha)
    assert forecast.value_at_risk > 0
    assert forecast.expected_shortfall > 0


@pytest.mark.parametrize("alpha", [0.10, 0.05, 0.01, 0.001])
def test_shortfall_is_never_below_value_at_risk(
    exact_sigma_returns: np.ndarray, alpha: float
) -> None:
    """The defining relationship between the two measures."""
    forecast = ParametricNormal().fit(exact_sigma_returns).forecast(alpha)
    assert forecast.expected_shortfall > forecast.value_at_risk


def test_forecast_carries_its_own_alpha(exact_sigma_returns: np.ndarray) -> None:
    """So a forecast remains interpretable once it is in a table of many."""
    forecast = ParametricNormal().fit(exact_sigma_returns).forecast(0.025)
    assert forecast.alpha == 0.025


def test_risk_increases_as_alpha_shrinks(exact_sigma_returns: np.ndarray) -> None:
    model = ParametricNormal().fit(exact_sigma_returns)
    figures = [model.forecast(a).value_at_risk for a in (0.10, 0.05, 0.01, 0.001)]
    assert figures == sorted(figures)


def test_risk_scales_linearly_with_volatility(exact_sigma_returns: np.ndarray) -> None:
    """Doubling every return doubles both figures. True of any scale family."""
    base = ParametricNormal().fit(exact_sigma_returns).forecast(0.01)
    doubled = ParametricNormal().fit(exact_sigma_returns * 2).forecast(0.01)

    assert doubled.value_at_risk == pytest.approx(2 * base.value_at_risk, rel=1e-12)
    assert doubled.expected_shortfall == pytest.approx(2 * base.expected_shortfall, rel=1e-12)


def test_fit_returns_self_for_chaining(exact_sigma_returns: np.ndarray) -> None:
    model = ParametricNormal()
    assert model.fit(exact_sigma_returns) is model


def test_parametric_normal_satisfies_the_protocol() -> None:
    """mypy checks this statically; this catches it at runtime too."""
    assert isinstance(ParametricNormal(), RiskModel)


def test_forecast_is_immutable(exact_sigma_returns: np.ndarray) -> None:
    """A forecast is a record of a result, not a mutable working value."""
    forecast = ParametricNormal().fit(exact_sigma_returns).forecast(0.01)
    with pytest.raises(AttributeError):
        forecast.value_at_risk = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The mean convention
# ---------------------------------------------------------------------------


def test_zero_mean_is_the_default() -> None:
    assert ParametricNormal().mean == "zero"


def test_sample_mean_shifts_the_forecast_by_exactly_the_mean() -> None:
    """The two settings differ by mu and nothing else."""
    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.0008, scale=0.012, size=500)
    mu = float(np.mean(returns))

    zero = ParametricNormal(mean="zero").fit(returns).forecast(0.01)
    sample = ParametricNormal(mean="sample").fit(returns).forecast(0.01)

    assert sample.value_at_risk == pytest.approx(zero.value_at_risk - mu, rel=1e-12)
    assert sample.expected_shortfall == pytest.approx(zero.expected_shortfall - mu, rel=1e-12)


def test_a_positive_mean_lowers_reported_risk() -> None:
    """Worth pinning: drift is subtracted from the loss, so it reduces VaR."""
    returns = np.full(300, 0.0)
    returns[:150], returns[150:] = -0.012, 0.012
    drifting = returns + 0.001

    zero = ParametricNormal(mean="zero").fit(drifting).forecast(0.01)
    sample = ParametricNormal(mean="sample").fit(drifting).forecast(0.01)
    assert sample.value_at_risk < zero.value_at_risk


def test_unknown_mean_setting_is_rejected() -> None:
    with pytest.raises(ValueError, match="'zero' or 'sample'"):
        ParametricNormal(mean="average")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Failure before nonsense
# ---------------------------------------------------------------------------


def test_forecasting_before_fitting_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        ParametricNormal().forecast(0.01)


@pytest.mark.parametrize("attribute", ["volatility", "n_observations"])
def test_diagnostics_before_fitting_raise(attribute: str) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        getattr(ParametricNormal(), attribute)


def test_too_few_observations_is_rejected() -> None:
    with pytest.raises(ValueError, match=f"at least {MIN_OBSERVATIONS}"):
        ParametricNormal().fit(np.zeros(MIN_OBSERVATIONS - 1))


def test_non_finite_observations_are_dropped(exact_sigma_returns: np.ndarray) -> None:
    """One NaN otherwise turns every downstream figure into NaN, silently."""
    dirty = np.concatenate([exact_sigma_returns, [np.nan, np.inf, -np.inf]])
    model = ParametricNormal().fit(dirty)

    assert model.n_observations == 250
    assert np.isfinite(model.volatility)


def test_bad_alpha_reports_the_existing_message(exact_sigma_returns: np.ndarray) -> None:
    """Validation is not duplicated here; it propagates from the distribution layer."""
    model = ParametricNormal().fit(exact_sigma_returns)
    with pytest.raises(ValueError, match="tail probability"):
        model.forecast(0.99)


def test_accepts_a_plain_list() -> None:
    """Callers should not have to convert to an array first."""
    model = ParametricNormal().fit([0.01, -0.01] * 50)
    assert model.n_observations == 100


# ---------------------------------------------------------------------------
# Consistency with the layer below
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.10, 0.05, 0.01, 0.001])
def test_forecast_equals_the_distribution_layer_negated(
    exact_sigma_returns: np.ndarray, alpha: float
) -> None:
    """Guards against a second, divergent sign flip creeping in later."""
    sigma = 0.012
    forecast = ParametricNormal().fit(exact_sigma_returns).forecast(alpha)

    assert forecast.value_at_risk == pytest.approx(
        -sigma * standard_normal_quantile(alpha), rel=1e-12
    )
    assert forecast.expected_shortfall == pytest.approx(
        -sigma * standard_normal_expected_shortfall(alpha), rel=1e-12
    )


def test_repr_is_informative() -> None:
    unfitted = repr(ParametricNormal())
    assert "unfitted" in unfitted

    fitted = repr(ParametricNormal().fit([0.01, -0.01] * 50))
    assert "sigma=" in fitted and "n=100" in fitted


def test_forecast_is_a_riskforecast(exact_sigma_returns: np.ndarray) -> None:
    forecast = ParametricNormal().fit(exact_sigma_returns).forecast(0.01)
    assert isinstance(forecast, RiskForecast)

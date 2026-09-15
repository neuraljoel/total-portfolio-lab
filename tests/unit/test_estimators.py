"""Tests for the estimators and the interface they share.

The estimators themselves are simple, so most of these tests are really about
the *contract* any estimator must honour: positive losses, ES at least as deep
as VaR, errors before nonsense, and linear scaling in volatility.

The two estimators are structurally opposite — one assumes a distribution and
stores two parameters, the other assumes nothing and stores the whole window —
so the contract holding for both is evidence that it generalises.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from tplab.risk import (
    MIN_OBSERVATIONS,
    HistoricalSimulation,
    ParametricNormal,
    RiskForecast,
    RiskModel,
)
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


# ===========================================================================
# HistoricalSimulation
# ===========================================================================


@pytest.fixture
def ramp() -> np.ndarray:
    """100 returns evenly spaced from -0.050 to 0.049, step 0.001.

    Chosen so the quantile and the tail average can both be worked out by hand:
    at alpha=0.05 the position is 0.05*99 = 4.95, sitting 95% of the way from
    -0.046 to -0.045, and the five observations below it average to -0.048.
    """
    return np.arange(-50, 50) / 1000.0


# --- the number itself -----------------------------------------------------


def test_var_is_the_empirical_quantile(ramp: np.ndarray) -> None:
    forecast = HistoricalSimulation().fit(ramp).forecast(alpha=0.05)
    assert forecast.value_at_risk == pytest.approx(0.045050, abs=1e-9)


def test_es_is_the_mean_of_the_tail(ramp: np.ndarray) -> None:
    """The five worst days are -0.050 .. -0.046, averaging -0.048."""
    forecast = HistoricalSimulation().fit(ramp).forecast(alpha=0.05)
    assert forecast.expected_shortfall == pytest.approx(0.048, abs=1e-9)


def test_es_matches_an_independent_tail_average(ramp: np.ndarray) -> None:
    """Recomputed from the raw data rather than trusting the implementation."""
    quantile = np.quantile(ramp, 0.05)
    expected = -float(ramp[ramp < quantile].mean())

    forecast = HistoricalSimulation().fit(ramp).forecast(alpha=0.05)
    assert forecast.expected_shortfall == pytest.approx(expected, rel=1e-12)


def test_var_never_exceeds_the_worst_observation(ramp: np.ndarray) -> None:
    """The method cannot reach past the data it was given."""
    forecast = HistoricalSimulation().fit(ramp).forecast(alpha=0.05)
    assert forecast.value_at_risk <= -float(ramp.min())


# --- the shared contract ---------------------------------------------------


@pytest.mark.parametrize("alpha", [0.20, 0.10, 0.05])
def test_hs_losses_are_positive(ramp: np.ndarray, alpha: float) -> None:
    forecast = HistoricalSimulation().fit(ramp).forecast(alpha)
    assert forecast.value_at_risk > 0
    assert forecast.expected_shortfall > 0


@pytest.mark.parametrize("alpha", [0.20, 0.10, 0.05])
def test_hs_shortfall_is_never_below_var(ramp: np.ndarray, alpha: float) -> None:
    forecast = HistoricalSimulation().fit(ramp).forecast(alpha)
    assert forecast.expected_shortfall > forecast.value_at_risk


def test_hs_risk_increases_as_alpha_shrinks(ramp: np.ndarray) -> None:
    model = HistoricalSimulation().fit(ramp)
    figures = [model.forecast(a).value_at_risk for a in (0.20, 0.10, 0.05)]
    assert figures == sorted(figures)


def test_hs_scales_linearly(ramp: np.ndarray) -> None:
    base = HistoricalSimulation().fit(ramp).forecast(0.05)
    doubled = HistoricalSimulation().fit(ramp * 2).forecast(0.05)

    assert doubled.value_at_risk == pytest.approx(2 * base.value_at_risk, rel=1e-12)
    assert doubled.expected_shortfall == pytest.approx(2 * base.expected_shortfall, rel=1e-12)


def test_hs_forecast_carries_its_alpha(ramp: np.ndarray) -> None:
    assert HistoricalSimulation().fit(ramp).forecast(0.05).alpha == 0.05


def test_hs_fit_returns_self(ramp: np.ndarray) -> None:
    model = HistoricalSimulation()
    assert model.fit(ramp) is model


def test_hs_satisfies_the_protocol() -> None:
    """The point of building this estimator second: the interface generalises."""
    assert isinstance(HistoricalSimulation(), RiskModel)


def test_hs_takes_no_configuration() -> None:
    """No mu + sigma*q decomposition, so nothing to configure."""
    with pytest.raises(TypeError):
        HistoricalSimulation("zero")  # type: ignore[call-arg]


# --- the thin-tail guard ---------------------------------------------------


def test_thin_tail_is_refused() -> None:
    """250 days at alpha=0.001 gives 0.25 expected tail observations."""
    window = np.random.default_rng(0).normal(0, 0.012, 250)
    model = HistoricalSimulation().fit(window)

    with pytest.raises(ValueError, match="is not an estimate"):
        model.forecast(0.001)


def test_thin_tail_message_names_the_cause() -> None:
    window = np.random.default_rng(0).normal(0, 0.012, 250)
    with pytest.raises(ValueError, match=r"n\*alpha = 0\.25"):
        HistoricalSimulation().fit(window).forecast(0.001)


def test_a_longer_window_makes_the_same_alpha_usable() -> None:
    """The guard is about sample size, not about alpha being small."""
    rng = np.random.default_rng(0)
    short, long = rng.normal(0, 0.012, 250), rng.normal(0, 0.012, 20_000)

    with pytest.raises(ValueError):
        HistoricalSimulation().fit(short).forecast(0.001)

    assert HistoricalSimulation().fit(long).forecast(0.001).value_at_risk > 0


# --- failure before nonsense -----------------------------------------------


def test_hs_forecasting_before_fitting_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        HistoricalSimulation().forecast(0.05)


def test_hs_n_observations_before_fitting_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = HistoricalSimulation().n_observations


def test_hs_too_few_observations_is_rejected() -> None:
    with pytest.raises(ValueError, match=f"at least {MIN_OBSERVATIONS}"):
        HistoricalSimulation().fit(np.zeros(MIN_OBSERVATIONS - 1))


def test_hs_bad_alpha_reports_the_shared_message(ramp: np.ndarray) -> None:
    """Validation comes from the distribution layer, not a second copy."""
    with pytest.raises(ValueError, match="tail probability"):
        HistoricalSimulation().fit(ramp).forecast(0.99)


def test_hs_drops_non_finite_observations(ramp: np.ndarray) -> None:
    dirty = np.concatenate([ramp, [np.nan, np.inf, -np.inf]])
    model = HistoricalSimulation().fit(dirty)
    assert model.n_observations == 100


def test_hs_repr_is_informative(ramp: np.ndarray) -> None:
    assert "unfitted" in repr(HistoricalSimulation())
    assert "n=100" in repr(HistoricalSimulation().fit(ramp))


# --- against the parametric model ------------------------------------------


def test_the_two_estimators_agree_on_normal_data() -> None:
    """With genuinely normal returns the distribution assumption costs nothing."""
    returns = np.random.default_rng(7).normal(0, 0.012, 200_000)

    empirical = HistoricalSimulation().fit(returns).forecast(0.01)
    parametric = ParametricNormal().fit(returns).forecast(0.01)

    assert empirical.value_at_risk == pytest.approx(parametric.value_at_risk, rel=0.03)


def test_historical_simulation_sees_fat_tails_that_the_normal_model_misses() -> None:
    """The reason for having both. Student-t data, same volatility, deeper tail."""
    rng = np.random.default_rng(11)
    returns = stats.t.rvs(df=3, size=200_000, random_state=rng) * 0.012 / np.sqrt(3.0)

    empirical = HistoricalSimulation().fit(returns).forecast(0.01)
    parametric = ParametricNormal().fit(returns).forecast(0.01)

    assert empirical.value_at_risk > parametric.value_at_risk

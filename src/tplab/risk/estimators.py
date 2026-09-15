r"""Tail risk estimators, behind a single interface.

Estimators differ in how they describe the next period's return distribution,
not in what they produce. All implement :class:`RiskModel` and return a
:class:`RiskForecast`, so code that consumes forecasts is written once and works
with any of them.

Losses are reported as **positive** numbers here, unlike in
:mod:`tplab.stats.distributions`. The sign flip happens only in ``forecast``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt

from tplab.stats.distributions import (
    check_alpha,
    standard_normal_expected_shortfall,
    standard_normal_quantile,
)

__all__ = [
    "MIN_OBSERVATIONS",
    "MIN_TAIL_OBSERVATIONS",
    "HistoricalSimulation",
    "ParametricNormal",
    "RiskForecast",
    "RiskModel",
]

#: Fewer observations than this and a volatility estimate is not worth having.
MIN_OBSERVATIONS = 30

#: Fewest observations an empirical tail average may be taken over.
MIN_TAIL_OBSERVATIONS = 2


@dataclass(frozen=True)
class RiskForecast:
    """A one-period-ahead tail risk estimate, as positive loss magnitudes.

    Attributes
    ----------
    alpha:
        Tail probability the forecast was made at. Carried so a forecast stays
        interpretable in a table of many.
    value_at_risk:
        Loss threshold expected to be breached a fraction ``alpha`` of the time.
    expected_shortfall:
        Mean loss given that breach. Never below ``value_at_risk``.
    """

    alpha: float
    value_at_risk: float
    expected_shortfall: float


@runtime_checkable
class RiskModel(Protocol):
    """What every estimator provides: ``fit(returns)``, then ``forecast(alpha)``.

    A :class:`typing.Protocol` — estimators conform by having these methods, not
    by inheriting. The two are separate because fitting is the expensive half:
    a rolling analysis refits once per window while querying several confidence
    levels against each fit.
    """

    def fit(self, returns: npt.ArrayLike) -> Self:
        """Estimate parameters from a window of returns. Returns ``self``."""
        ...

    def forecast(self, alpha: float) -> RiskForecast:
        """One-period-ahead forecast at tail probability ``alpha``."""
        ...


def _clean(returns: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Flatten to 1-D float and drop non-finite values.

    Dropped rather than propagated: one NaN otherwise turns every downstream
    figure into NaN, surfacing far from its cause.
    """
    x = np.asarray(returns, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]

    if x.size < MIN_OBSERVATIONS:
        raise ValueError(
            f"need at least {MIN_OBSERVATIONS} finite observations to estimate "
            f"a volatility, got {x.size}"
        )
    return x


class ParametricNormal:
    r"""VaR and Expected Shortfall assuming returns are normally distributed.

    .. math::
        \mathrm{VaR}_\alpha &= -\left(\mu + \sigma\,\Phi^{-1}(\alpha)\right) \\
        \mathrm{ES}_\alpha  &= -\left(\mu + \sigma\,e(\alpha)\right)

    The standard baseline. Its two assumptions — thin tails and constant
    variance — are both false for real returns, so it tends to understate risk
    in turbulent periods.

    Parameters
    ----------
    mean:
        ``"zero"`` (default) sets :math:`\mu = 0`; ``"sample"`` uses the window
        mean. Zero is standard at daily frequency, where the mean return is not
        estimable to useful precision and including it adds noise.

    Examples
    --------
    >>> import numpy as np
    >>> returns = np.concatenate([np.full(125, -0.012), np.full(125, 0.012)])
    >>> model = ParametricNormal().fit(returns)
    >>> forecast = model.forecast(alpha=0.01)
    >>> f"{forecast.value_at_risk:.5f}", f"{forecast.expected_shortfall:.5f}"
    ('0.02797', '0.03205')
    """

    def __init__(self, mean: Literal["zero", "sample"] = "zero") -> None:
        if mean not in ("zero", "sample"):
            raise ValueError(f"mean must be 'zero' or 'sample', got {mean!r}")
        self.mean = mean
        self._mu: float | None = None
        self._sigma: float | None = None
        self._n_observations: int | None = None

    def fit(self, returns: npt.ArrayLike) -> Self:
        """Estimate the mean and volatility of the supplied window."""
        x = _clean(returns)
        self._mu = float(np.mean(x)) if self.mean == "sample" else 0.0
        self._sigma = float(np.std(x, ddof=1))
        self._n_observations = int(x.size)
        return self

    def forecast(self, alpha: float) -> RiskForecast:
        """One-period-ahead VaR and ES. Raises if unfitted or ``alpha`` is invalid."""
        if self._mu is None or self._sigma is None:
            raise RuntimeError("model is not fitted; call fit(returns) first")

        quantile = standard_normal_quantile(alpha)
        shortfall = standard_normal_expected_shortfall(alpha)

        return RiskForecast(
            alpha=alpha,
            value_at_risk=-(self._mu + self._sigma * quantile),
            expected_shortfall=-(self._mu + self._sigma * shortfall),
        )

    @property
    def volatility(self) -> float:
        """The fitted volatility."""
        if self._sigma is None:
            raise RuntimeError("model is not fitted; call fit(returns) first")
        return self._sigma

    @property
    def n_observations(self) -> int:
        """Number of finite observations the fit used."""
        if self._n_observations is None:
            raise RuntimeError("model is not fitted; call fit(returns) first")
        return self._n_observations

    def __repr__(self) -> str:
        if self._sigma is None:
            return f"ParametricNormal(mean={self.mean!r}, unfitted)"
        return (
            f"ParametricNormal(mean={self.mean!r}, sigma={self._sigma:.6f}, "
            f"n={self._n_observations})"
        )


class HistoricalSimulation:
    r"""VaR and Expected Shortfall read directly off the observed returns.

    No distribution is assumed. VaR is the empirical :math:`\alpha`-quantile of
    the window; ES is the mean of the observations strictly below it.

    The appeal is that whatever skew and fat tails the history contains are kept
    exactly as they occurred. The cost is that every forecast is a statement
    about days already seen: the estimate cannot reach past the worst observation
    in the window, and it moves in steps as extreme days enter and leave it.

    Interpolation between order statistics uses ``numpy``'s default linear
    method. With 250 observations at :math:`\alpha = 0.01` the quantile falls
    between the second and third worst returns, so the choice is not cosmetic.

    Examples
    --------
    >>> import numpy as np
    >>> returns = np.arange(-50, 50) / 1000.0
    >>> forecast = HistoricalSimulation().fit(returns).forecast(alpha=0.05)
    >>> f"{forecast.value_at_risk:.5f}", f"{forecast.expected_shortfall:.5f}"
    ('0.04505', '0.04800')
    """

    def __init__(self) -> None:
        self._window: npt.NDArray[np.float64] | None = None

    def fit(self, returns: npt.ArrayLike) -> Self:
        """Store the window. There is nothing to estimate."""
        self._window = _clean(returns)
        return self

    def forecast(self, alpha: float) -> RiskForecast:
        """One-period-ahead VaR and ES from the empirical distribution.

        Raises ``RuntimeError`` if unfitted, and ``ValueError`` if ``alpha`` is
        invalid or the window is too short to populate the tail.
        """
        if self._window is None:
            raise RuntimeError("model is not fitted; call fit(returns) first")
        check_alpha(alpha)

        quantile = float(np.quantile(self._window, alpha))
        tail = self._window[self._window < quantile]

        if tail.size < MIN_TAIL_OBSERVATIONS:
            n = self._window.size
            raise ValueError(
                f"only {tail.size} observation(s) fall below the empirical "
                f"{alpha:.3%} quantile of a {n}-observation window, and an average "
                f"over fewer than {MIN_TAIL_OBSERVATIONS} is not an estimate. "
                f"A window holds roughly n*alpha = {n * alpha:.2f} tail observations; "
                f"lengthen the window or raise alpha. Estimating beyond what the "
                f"sample contains is what extreme value theory is for."
            )

        return RiskForecast(
            alpha=alpha,
            value_at_risk=-quantile,
            expected_shortfall=-float(tail.mean()),
        )

    @property
    def n_observations(self) -> int:
        """Number of finite observations in the window."""
        if self._window is None:
            raise RuntimeError("model is not fitted; call fit(returns) first")
        return int(self._window.size)

    def __repr__(self) -> str:
        if self._window is None:
            return "HistoricalSimulation(unfitted)"
        return f"HistoricalSimulation(n={self._window.size})"

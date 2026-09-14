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
    standard_normal_expected_shortfall,
    standard_normal_quantile,
)

__all__ = [
    "MIN_OBSERVATIONS",
    "ParametricNormal",
    "RiskForecast",
    "RiskModel",
]

#: Fewer observations than this and a volatility estimate is not worth having.
MIN_OBSERVATIONS = 30


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

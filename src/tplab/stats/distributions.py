r"""Standardised distribution quantiles and tail expectations.

Distribution *shapes* only. Nothing here sees a return series, a mean, or a
volatility — those belong to :mod:`tplab.risk`.

Notation: ``phi(x)`` is the standard normal density, ``Phi(x)`` its cdf, and
``Phi^-1(alpha)`` its quantile function (``norm.pdf``, ``norm.cdf``, ``norm.ppf``
in scipy).

Conventions
-----------
``alpha`` is the **tail probability**, not the confidence level: 99% VaR uses
``alpha=0.01``. Values at or above 0.5 are rejected, so a confidence level passed
by mistake fails loudly instead of silently returning ``+2.33``.

Results are **standardised** — zero mean, unit variance — and therefore negative.
This lets a parametric estimate factorise as ``VaR = mu + sigma * q(alpha)``,
where ``q`` depends only on distribution shape. Converting to a positive loss
happens once, in :mod:`tplab.risk`.
"""

from __future__ import annotations

from scipy import stats

__all__ = [
    "MAX_ALPHA",
    "MIN_ALPHA",
    "check_alpha",
    "standard_normal_expected_shortfall",
    "standard_normal_quantile",
]

#: Tail probabilities must lie strictly inside this interval. The upper bound is
#: 0.5 rather than 1.0 so that a confidence level passed by mistake is caught.
MIN_ALPHA = 0.0
MAX_ALPHA = 0.5


def check_alpha(alpha: float) -> None:
    """Reject anything that is not a usable loss-tail probability.

    Public because the convention is shared: estimators that do not route
    through a quantile function still have to enforce it, and one message for
    one mistake is worth more than a private helper.
    """
    if not MIN_ALPHA < alpha < MAX_ALPHA:
        raise ValueError(
            f"alpha must be a tail probability in ({MIN_ALPHA}, {MAX_ALPHA}), got {alpha}. "
            f"For 99% VaR pass alpha=0.01, not 0.99."
        )


def standard_normal_quantile(alpha: float) -> float:
    r"""The value below which a fraction ``alpha`` of the standard normal sits.

    .. math::  x_\alpha = \Phi^{-1}(\alpha)

    Scaled by a volatility, this is parametric normal Value at Risk.

    Examples
    --------
    >>> round(standard_normal_quantile(0.05), 4)
    -1.6449
    >>> round(standard_normal_quantile(0.01), 4)
    -2.3263
    """
    check_alpha(alpha)
    return float(stats.norm.ppf(alpha))


def standard_normal_expected_shortfall(alpha: float) -> float:
    r"""The average value in the tail below the ``alpha`` quantile.

    Where the quantile marks the edge of the tail, this describes what lies
    beyond it — the mean loss *given* that VaR was breached.

    .. math::
        \mathrm{ES}_\alpha
            = \frac{1}{\alpha}\int_{-\infty}^{x_\alpha} x\,\varphi(x)\,dx
            = -\frac{\varphi(x_\alpha)}{\alpha}

    The closed form follows because :math:`\varphi'(x) = -x\,\varphi(x)`, so the
    integrand is the derivative of :math:`-\varphi(x)`. scipy has no function for
    this. Always strictly below the quantile.

    Examples
    --------
    >>> round(standard_normal_expected_shortfall(0.05), 4)
    -2.0627
    >>> round(standard_normal_expected_shortfall(0.01), 4)
    -2.6652
    """
    check_alpha(alpha)
    x_alpha = stats.norm.ppf(alpha)
    return float(-stats.norm.pdf(x_alpha) / alpha)

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from ..misc import find_closest
from .density import density


def hdi(x, ci=0.95, show=False, **kwargs):
    """**Highest Density Interval (HDI)**

    Compute the Highest Density Interval (HDI) of a distribution. All points within this interval
    have a higher probability density than points outside the interval. The HDI can be used in the
    context of uncertainty characterisation of posterior distributions (in the Bayesian farmework)
    as Credible Interval (CI). Unlike equal-tailed intervals that typically exclude 2.5% from each
    tail of the distribution and always include the median, the HDI is not equal-tailed and
    therefore always includes the mode(s) of posterior distributions.

    Parameters
    ----------
    x : Union[list, np.array, pd.Series]
        A vector of values.
    ci : float
        Value of probability of the (credible) interval - CI (between 0 and 1) to be estimated.
        Default to .95 (95%).
    show : bool
        If ``True``, the function will produce a figure.
    **kwargs : Line2D properties
        Other arguments to be passed to :func:`nk.density`.

    See Also
    --------
    density

    Returns
    ----------
    float(s)
        The HDI low and high limits.
    fig
        Distribution plot.

    Examples
    ----------
    .. ipython:: python

      import numpy as np
      import neurokit2 as nk

      x = np.random.normal(loc=0, scale=1, size=100000)
      @savefig p_hdi1.png scale=100%
      ci_min, ci_high = nk.hdi(x, ci=0.95, show=True)
      @suppress
      plt.close()

    """
    x_sorted = np.sort(x)
    window_size = np.ceil(ci * len(x_sorted)).astype("int")

    if window_size < 2:
        raise ValueError(
            "NeuroKit error: hdi(): `ci` is too small or x does not contain enough data points."
        )

    nCIs = len(x_sorted) - window_size

    ciWidth = [0] * nCIs
    for i in np.arange(0, nCIs):
        ciWidth[i] = x_sorted[i + window_size] - x_sorted[i]
    hdi_low = x_sorted[ciWidth.index(np.min(ciWidth))]
    hdi_high = x_sorted[ciWidth.index(np.min(ciWidth)) + window_size]

    if show is True:
        _hdi_plot(x, hdi_low, hdi_high, **kwargs)

    return hdi_low, hdi_high


def _hdi_plot(vals, hdi_low, hdi_high, ci=0.95, **kwargs):
    x, y = density(vals, show=False, **kwargs)

    where = np.full(len(x), False)
    where[0 : find_closest(hdi_low, x, return_index=True)] = True
    where[find_closest(hdi_high, x, return_index=True) : :] = True

    fig, ax = plt.subplots()  # pylint: disable=unused-variable
    ax.plot(x, y, color="white")
    ax.fill_between(
        x,
        y,
        where=where,
        color="#E91E63",
        label="CI {:.0%} [{:.2f}, {:.2f}]".format(ci, hdi_low, hdi_high),
    )
    ax.fill_between(x, y, where=~where, color="#2196F3")
    ax.legend(loc="upper right")

# -*- coding: utf-8 -*-
import numpy as np




def hdi(x, ci=0.95):
    """Highest Density Interval (HDI)

    Compute the Highest Density Interval (HDI) of a distribution. All points within this interval have a higher probability density than points outside the interval. The HDI can be used in the context of uncertainty characterisation of posterior distributions (in the Bayesian farmework) as Credible Interval (CI). Unlike equal-tailed intervals that typically exclude 2.5% from each tail of the distribution and always include the median, the HDI is not equal-tailed and therefore always includes the mode(s) of posterior distributions.

    Parameters
    ----------
    x : list, array or Series
        A vector of values.
    ci : float
        Value of probability of the (credible) interval - CI (between 0 and 1) to be estimated. Default to .95 (95%).

    Returns
    ----------
    float, floats
        The HDI low and high limits.


    Examples
    ----------
    >>> import numpy as np
    >>> import neurokit2 as nk
    >>>
    >>> x = np.random.normal(loc=0, scale=1, size=1000000)
    >>> ci_min, ci_max = nk.hdi(x, ci=0.95)
    """
    x_sorted = np.sort(x)
    window_size = np.ceil(ci * len(x_sorted)).astype('int')

    if window_size < 2:
        raise ValueError("NeuroKit error: hdi(): `ci` is too small or x does not contain enough data points.")

    nCIs = len(x_sorted) - window_size

    ciWidth = [0]*nCIs
    for i in np.arange(0, nCIs):
        ciWidth[i] = x_sorted[i + window_size] - x_sorted[i]
    hdi_low = x_sorted[ciWidth.index(np.min(ciWidth))]
    hdi_max = x_sorted[ciWidth.index(np.min(ciWidth))+window_size]

    return hdi_low, hdi_max

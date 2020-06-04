# -*- coding: utf-8 -*-
import numpy as np


def mad(x, constant=1.4826):
    """
    Median Absolute Deviation: a "robust" version of standard deviation.

    Parameters
    ----------
    x : list, array or Series
        A vector of values.
    constant : float
        Scale factor. Use 1.4826 for results similar to default R.

    Returns
    ----------
    mad : float
        The MAD.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>> nk.mad([2, 8, 7, 5, 4, 12, 5, 1])
    3.7064999999999997

    References
    -----------
    - https://en.wikipedia.org/wiki/Median_absolute_deviation

    """
    median = np.nanmedian(np.ma.array(x).compressed())
    mad = np.nanmedian(np.abs(x - median))
    mad = mad * constant
    return mad

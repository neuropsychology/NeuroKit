# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .mad import mad


def standardize(data, robust=False):
    """
    Standardization of data.

    Performs a standardization of data (Z-scoring), i.e., centering and scaling,
    so that the data is expressed in terms of standard deviation
    (i.e., mean = 0, SD = 1) or Median Absolute Deviance (median = 0, MAD = 1).

    Parameters
    ----------
    data : list, array or Series
        Raw data.
    robust : bool
        If True, centering is done by substracting the median from the
        variables and dividing it by the median absolute deviation (MAD).
        If False, variables are standardized by substracting the mean and
        dividing it by the standard deviation (SD).

    Returns
    ----------
    list, array or Series
        The standardized values.


    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> x = nk.standardize([3, 1, 2, 4, 6, np.nan])
    >>> y = nk.standardize([3, 1, 2, 4, 6, np.nan], robust=True)
    >>> z = nk.standardize(pd.DataFrame({"A": [3, 1, 2, 4, 6, np.nan], "B": [3, 1, 2, 4, 6, 5]}))
    >>> z #doctest: +SKIP

    """
    # Return appropriate type
    if isinstance(data, list):
        data = list(_standardize(np.array(data), robust=robust))
    else:
        data = _standardize(data, robust=robust)

    return data


def _standardize(data, robust=False):

    # Compute standardized
    if robust is False:
        z = (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0, ddof=1)
    else:
        z = (data - np.nanmedian(data, axis=0)) / mad(data)

    return z

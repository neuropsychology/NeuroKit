# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import scipy.stats








def _standardize(data, robust=False):

    if robust is False:
        z = (data - np.mean(data, axis=0))/np.std(data, axis=0)
    else:
        z = (data - np.median(data, axis=0))/scipy.stats.median_absolute_deviation(data, axis=0, nan_policy="omit")

    return(z)









def standardize(data, robust=False):
    """Standardization of data

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
    >>> nk.standardize([3, 1, 2, 4, 6])
    """
     # Return appropriate type
    if isinstance(data, list):
        data = list(_standardize(np.array(data), robust=robust))
    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        z = _standardize(data.values, robust=robust)
        data[:] = z
    else:
        data = _standardize(data, robust=robust)

    return(data)




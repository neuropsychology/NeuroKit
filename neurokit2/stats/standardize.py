# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .mad import mad


def standardize(data, robust=False, window=None, **kwargs):
    """Standardization of data.

    Performs a standardization of data (Z-scoring), i.e., centering and scaling, so that the data is
    expressed in terms of standard deviation (i.e., mean = 0, SD = 1) or Median Absolute Deviance
    (median = 0, MAD = 1).

    Parameters
    ----------
    data : Union[list, np.array, pd.Series]
        Raw data.
    robust : bool
        If True, centering is done by substracting the median from the variables and dividing it by
        the median absolute deviation (MAD). If False, variables are standardized by substracting the
        mean and dividing it by the standard deviation (SD).
    window : int
        Perform a rolling window standardization, i.e., apply a standardization on a window of the
        specified number of samples that rolls along the main axis of the signal. Can be used for
        complex detrending.
    **kwargs : optional
        Other arguments to be passed to ``pandas.rolling()``.

    Returns
    ----------
    list
        The standardized values.


    Examples
    ----------
    >>> import neurokit2 as nk
    >>> import pandas as pd
    >>>
    >>> # Simple example
    >>> nk.standardize([3, 1, 2, 4, 6, np.nan]) #doctest: +ELLIPSIS
    [...]
    >>> nk.standardize([3, 1, 2, 4, 6, np.nan], robust=True) #doctest: +ELLIPSIS
    [...]
    >>> nk.standardize(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T) #doctest: +ELLIPSIS
     array(...)
    >>> nk.standardize(pd.DataFrame({"A": [3, 1, 2, 4, 6, np.nan],
    ...                              "B": [3, 1, 2, 4, 6, 5]})) #doctest: +ELLIPSIS
              A         B
    0       ...       ...
    ...
    >>>
    >>> # Rolling standardization of a signal
    >>> signal = nk.signal_simulate(frequency=[0.1, 2], sampling_rate=200)
    >>> z = nk.standardize(signal, window=200)
    >>> nk.signal_plot([signal, z], standardize=True)

    """
    # Return appropriate type
    if isinstance(data, list):
        data = list(_standardize(np.array(data), robust=robust, window=window, **kwargs))
    elif isinstance(data, pd.DataFrame):
        data = pd.DataFrame(_standardize(data, robust=robust, window=window, **kwargs))
    elif isinstance(data, pd.Series):
        data = pd.Series(_standardize(data, robust=robust, window=window, **kwargs))
    else:
        data = _standardize(data, robust=robust, window=window, **kwargs)

    return data


# =============================================================================
# Internals
# =============================================================================
def _standardize(data, robust=False, window=None, **kwargs):

    # Compute standardized on whole data
    if window is None:
        if robust is False:
            z = (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0, ddof=1)
        else:
            z = (data - np.nanmedian(data, axis=0)) / mad(data)

    # Rolling standardization on windows
    else:
        df = pd.DataFrame(data)  # Force dataframe

        if robust is False:
            z = (df - df.rolling(window, min_periods=0, **kwargs).mean()) / df.rolling(
                window, min_periods=0, **kwargs
            ).std(ddof=1)
        else:
            z = (df - df.rolling(window, min_periods=0, **kwargs).median()) / df.rolling(
                window, min_periods=0, **kwargs
            ).apply(mad)

        # Fill the created nans
        z = z.fillna(method="bfill")

        # Restore to vector or array
        if z.shape[1] == 1:
            z = z[0].values
        else:
            z = z.values

    return z

import numpy as np


def mad(x, constant=1.4826, **kwargs):
    """**Median Absolute Deviation: a "robust" version of standard deviation**

    Parameters
    ----------
    x : Union[list, np.array, pd.Series]
        A vector of values.
    constant : float
        Scale factor. Use 1.4826 for results similar to default R.

    Returns
    ----------
    float
        The MAD.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk
      nk.mad([2, 8, 7, 5, 4, 12, 5, 1])


    References
    -----------
    * https://en.wikipedia.org/wiki/Median_absolute_deviation

    """
    if kwargs.get("axis") is None:
        median = np.nanmedian(np.ma.array(x).compressed(), **kwargs)
    else:
        # Center each slice on its own median, not on that of the flattened data
        median = np.nanmedian(x, axis=kwargs["axis"], keepdims=True)
    mad_value = np.nanmedian(np.abs(x - median), **kwargs)
    mad_value = mad_value * constant
    return mad_value

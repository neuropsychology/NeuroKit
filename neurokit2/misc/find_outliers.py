import numpy as np
import scipy
from ..stats import standardize

def find_outliers(data, exclude=5, tail="two-tail"):
    """
    Identify outliers (abnormal values).

    Parameters
    ----------
    data : list or ndarray
        Data array
    exclude : int, float
        Percentage of extreme observation to be excluded.
    tail: str
        Can be "two-tail", "left-tail" or "right-tail. If exclude=5 and tail="two-tail", 2.5% of extreme
        observation of each tail will be marked as outliers.

    Returns
    ----------
    outliers : ndarray
        A list of True/False with True being the outliers.

    Example
    ----------
    >>> import neurokit2 as nk
    >>> outliers = nk.find_outliers([1, 2, 1, 5, 666, 4, 1 ,3, 5])
    """

    z = np.array(standardize(data))
    if tail == "two-tail":
        exclude = exclude / 2
        outliers = abs(z) > scipy.stats.norm.ppf(1 - exclude / 100)
    if tail == "left-tail":
        outliers = z < -scipy.stats.norm.ppf(1 - exclude / 100)
    if tail == "right-tail":
        outliers = z > scipy.stats.norm.ppf(1 - exclude / 100)
    outliers = np.array(outliers)
    return outliers

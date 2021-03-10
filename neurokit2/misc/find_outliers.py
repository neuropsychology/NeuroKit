import numpy as np
import scipy
from ..stats import standardize

def find_outliers(data, exclude=0.05, side="both"):
    """
    Identify outliers (abnormal values).

    Parameters
    ----------
    data : list or ndarray
        Data array
    exclude : int, float
        Proportion of extreme observation to be excluded.
    side: str
        Can be "both", "left" or "right". If exclude=0.05 and side="both", 2.5% of extreme
        observation of each side will be marked as outliers.

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
    if side == "both":
        outliers = abs(z) > scipy.stats.norm.ppf(1 - (exclude / 2))
    elif side == "left":
        outliers = z < -scipy.stats.norm.ppf(1 - exclude)
    elif side == "right":
        outliers = z > scipy.stats.norm.ppf(1 - exclude)
    else:
        raise ValueError("side must be 'both', 'left' or 'right'.")

    outliers = np.array(outliers)
    return outliers

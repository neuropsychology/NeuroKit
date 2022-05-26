import numpy as np
import scipy

from ..stats import standardize


def find_outliers(data, 
                  exclude=0.05, 
                  side="both", 
                  method="standardize",
                  **kwargs):
    """Identify outliers (abnormal values)

    Extreme values identification.

    Parameters
    ----------
    data : list or ndarray
        Data array
    exclude : int, float
        Proportion of extreme observations to be excluded for "standardize" method.
    side: str
        Can be "both", "left" or "right". If side="both", the extreme
        values identified on each side will be marked as outliers.
    method: str
        Can be "standardize" or "percentile". The default is "standardize".
    **kwargs : optional
        Other arguments to be passed to :func:`_find_outliers_standardize` or
        :func:`_find_outliers_percentiles`.

    Returns
    ----------
    outliers : ndarray
        A list of True/False with True being the outliers.

    Example
    ----------
    >>> import neurokit2 as nk
    >>> outliers = nk.find_outliers([1, 2, 1, 5, 666, 4, 1 ,3, 5])
    >>> outliers
    array([False, False, False, False,  True, False, False, False, False])
    """
    if side not in ["both", "left", "right"]:
        raise ValueError("side must be 'both', 'left' or 'right'.")
        
    if method not in ["standardize", "percentile"]:
        raise ValueError("method must be 'standardize' or 'percentile'.")
        
    if method == "standardize":
        outliers = _find_outliers_standardize(data,
                                         exclude=exclude, 
                                         side=side,
                                         **kwargs)
    elif method == "percentile":
        outliers = _find_outliers_percentiles(data, side=side, **kwargs)
    return np.array(outliers)


def _find_outliers_standardize(data, exclude=0.05, side="both", **kwargs):
    """Identify outliers (abnormal values) with standardization

    Extreme values identification using standardization, i.e., centering and 
    scaling.

    Parameters
    ----------
    data : list or ndarray
        Data array
    exclude : int, float
        Proportion of extreme observation to be excluded.
    side: str
        Can be "both", "left" or "right". If exclude=0.05 and side="both", 
        2.5% of extreme observations of each side will be marked as outliers.
    **kwargs : optional
    Other arguments to be passed to :func:`standardize`.

    Returns
    ----------
    outliers : ndarray
        A list of True/False with True being the outliers.
    """
    if side not in ["both", "left", "right"]:
        raise ValueError("side must be 'both', 'left' or 'right'.")
        
    z = np.array(standardize(data, **kwargs))
    if side == "both":
        outliers = abs(z) > scipy.stats.norm.ppf(1 - (exclude / 2))
    elif side == "left":
        outliers = z < -scipy.stats.norm.ppf(1 - exclude)
    elif side == "right":
        outliers = z > scipy.stats.norm.ppf(1 - exclude)
    return np.array(outliers)

def _find_outliers_percentiles(data, 
                               side="both", 
                               percentile_threshold=(0.25, 0.75), 
                               percentile_range=(0.25, 0.75),
                               multiplier_range=1.5):
    """Identify outliers (abnormal values) with percentiles

    Extreme values identification using percentiles.

    Parameters
    ----------
    data : list or ndarray
        Data array
    side: str
        Can be "both", "left" or "right". If side="both", the extreme
        values identified on each side will be marked as outliers.
    percentile_threshold : tuple
        Percentiles for the left and right threshold.
    percentile_range : tuple
        Percentiles for the inter-percentile range. (0.25, 0.75) is the 
        interquartile range; (0.10, 0.90) is the interdecile range.
    multiplier_range : int, float
        Multiplier for the inter-percentile range. If side="both" and
        multiplier_range=1.5, observations more than 1.5 inter-percentile 
        ranges below the left percentile or above the right percentile will
        be marked as outliers.

    Returns
    ----------
    outliers : ndarray
        A list of True/False with True being the outliers.
    """
    if side not in ["both", "left", "right"]:
        raise ValueError("side must be 'both', 'left' or 'right'.")
        
    if type(data) is list:
        data = np.array(data)
        
    inter_percentile_range = (np.percentile(data, percentile_range[1]) - 
                   np.percentile(data, percentile_range[0]))
    left_threshold = (np.percentile(data, percentile_threshold[0]) - 
                      multiplier_range*inter_percentile_range)
    right_threshold = (np.percentile(data, percentile_threshold[1]) + 
                      multiplier_range*inter_percentile_range)
    if side == "both":
        outliers = (data > left_threshold) & (data < right_threshold)
    elif side == "left":
        outliers = data > left_threshold
    elif side == "right":
        outliers = data < right_threshold
    return outliers
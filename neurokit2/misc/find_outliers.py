import numpy as np
import scipy

from ..stats import standardize


def find_outliers(data, exclude=2, side="both", method="sd", **kwargs):
    """**Identify outliers (abnormal values)**

    Extreme values identification using different methods, such as:

    * **sd**: Data is :func:`standardized <.standardize>`, i.e., centered and
      scaled, and absolute value beyond a certain SD threshold are considered as outliers.
    * **norm**: Extreme values identified using theoretical percentiles to identify outliers
      beyond a certain theoretical percentile (assuming the data comes from a normal distribution).
      For example, with this method, ``exclude=0.025`` (one-sided) corresponds to the 2.5% lower
      bound of the normal distribution, which corresponds to approx. -1.96 SD. This method is
      related to the **SD** one, but instead of specifying the threshold in SDs, it is specified in
      percentiles.
    * **percentile**: Extreme values identified using percentiles.

    Parameters
    ----------
    data : list or ndarray
        Data array
    exclude : int, float
        Amount of outliers to detect (depends on the chosen method).
    side: str
        Can be ``"both"``, ``"left"`` or ``"right"``. If ``exclude=0.05`` and ``side="both"`` and
        ``method="norm"``, 2.5% of extreme observation of each side will be marked as outliers.
    method: str
        Can be "standardize" or "percentile". The default is "standardize".
    **kwargs : optional
        Other arguments to be passed to :func:`standardize`.

    Returns
    ----------
    outliers : ndarray
        A boolean vector of with ``True`` being the outliers.

    See Also
    ----------
    .standardize

    Example
    ----------
    .. ipython:: python

      import neurokit2 as nk

      data = [-12, 2, 1, 3, 66.6, 2, 1, 3, 2, -42, 2, 4, 1, 12]

      # Outliers beyond 2 SD of the mean
      outliers = nk.find_outliers(data, exclude=2, side="both", method="sd")
      np.where(outliers)[0]

      # Outliers beyond 1 MAD of the Median on one side
      outliers = nk.find_outliers(data, exclude=1, side="left", method="sd", robust=True)
      np.where(outliers)[0]

      # 2.5% theoretical percentiles on each side
      outliers = nk.find_outliers(data, exclude=0.05, method="norm")
      np.where(outliers)[0]

      # Outliers are beyond interquartile range
      outliers = nk.find_outliers(data, exclude=(0.25, 0.75), method="percentile")
      np.where(outliers)[0]

      # Outliers are beyond interdecile range
      outliers = nk.find_outliers(data, exclude=(0.1, 0.9), method="percentile")
      np.where(outliers)[0]

    """
    # Sanity checks
    if side not in ["both", "left", "right"]:
        raise ValueError("side must be 'both', 'left' or 'right'.")

    method = method.lower()
    if method not in ["standardize", "z", "sd", "percentile", "norm"]:
        raise ValueError("method must be 'standardize' or 'percentile'.")

    # Force array
    data = np.array(data)

    # Find thresholds
    if method in ["percentile"]:
        if isinstance(exclude, (list, tuple, np.ndarray)):
            right = np.percentile(data, exclude[1] * 100)
            left = np.percentile(data, exclude[0] * 100)
        else:
            right = np.percentile(data, (1 - (exclude / 2)) * 100)
            left = np.percentile(data, (exclude / 2) * 100)

    elif method in ["sd"]:
        if isinstance(exclude, (list, tuple, np.ndarray)):
            right = exclude[1]
            left = exclude[0]
        else:
            right = exclude
            left = -right
    else:
        if side == "both":
            exclude = exclude / 2
        right = scipy.stats.norm.ppf(1 - exclude)
        left = -right

    if method in ["standardize", "z", "sd", "norm"]:
        data = np.array(standardize(data, **kwargs))

    if side == "both":
        outliers = (data < left) | (data > right)
    elif side == "left":
        outliers = data < left
    elif side == "right":
        outliers = data > right

    return outliers

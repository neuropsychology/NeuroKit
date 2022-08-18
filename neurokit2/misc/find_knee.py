import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from ..stats import rescale


def find_knee(y, x=None, S=1, show=False, verbose=True):
    """**Find Knee / Elbow**

    Find the knee / elbow in a curve using a basic adaptation of the *kneedle* algorithm.

    Parameters
    ----------
    x : list
        A vector of values for which to detect the knee / elbow.
    S : float
        The sensitivity parameter allows us to adjust how aggressive we want to be when
        detecting knees. Smaller values detect knees quicker, while larger values are more
        conservative.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      y = np.log(np.arange(1, 100))
      y += np.random.normal(0, 0.2, len(y))
      nk.find_knee(y, show=True)

    References
    -----------
    * Satopaa, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011, June). Finding a" kneedle" in a
      haystack: Detecting knee points in system behavior. In 2011 31st international conference on
      distributed computing systems workshops (pp. 166-171). IEEE.

    """
    n = len(y)
    if n <= 5:
        raise ValueError("Input vector must have at least six values.")

    if x is None:
        x = np.arange(n)
    idx = rescale(x, to=[0, 1])

    # Smooth using spline
    spline = scipy.interpolate.UnivariateSpline(x=idx, y=y, k=5)
    smoothed = spline(idx)

    # Normalize to the unit square (0 - 1)
    smoothed = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))

    Y_d = smoothed - idx

    X_lm = []
    Y_lm = []

    maxima_ids = []
    for i in range(1, n - 1):
        if Y_d[i] > Y_d[i - 1] and Y_d[i] > Y_d[i + 1]:
            X_lm.append(idx[i])
            Y_lm.append(Y_d[i])
            maxima_ids.append(i)
    T_lm = Y_lm - S * np.sum(np.diff(idx)) / (n - 1)

    knee_point_index = _locate(Y_d, T_lm, maxima_ids)
    # If no knee point was found, return the last point
    if knee_point_index is None:
        if verbose is True:
            warnings.warn("No knee point found, retuning last.")
        knee = n - 1
    else:
        knee_point = X_lm[knee_point_index]
        # Which index
        knee = np.where(idx == knee_point)[0][0]

    if show is True:
        plt.plot(x, y, label="Original")
        plt.plot(x, rescale(smoothed, to=[np.nanmin(y), np.nanmax(y)]), label="Smoothed")
        plt.plot(x, rescale(Y_d, to=[np.nanmin(y), np.nanmax(y)]), label="Difference")
        plt.axvline(x=x[knee], color="red", linestyle="--")
        plt.legend()
    return x[knee]


def _locate(Y_d, T_lm, maxima_ids):
    n = len(Y_d)
    for j in range(0, n):
        for index, i in enumerate(maxima_ids):
            if j <= i:
                continue
            if Y_d[j] <= T_lm[index]:
                return index

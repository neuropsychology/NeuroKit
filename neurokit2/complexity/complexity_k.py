from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning


def complexity_k(signal, k_max="default", show=False):
    """
    The optimal kmax is computed based on the point at which HFD values plateau for a range of kmax values (see Vega, 2015).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series, np.ndarray, pd.DataFrame]
        The signal (i.e., a time series) in the form of a vector of values or in
        the form of an n-dimensional array (with a shape of len(channels) x len(samples))
        or dataframe.
    k_max : Union[int, str, list], optional
        Maximum number of interval times (should be greater than or equal to 3) to be tested. If 'default', it selects the maximum possible value corresponding to half the length of the signal.
    show : bool
        Visualise the slope of the curve for the selected kmax value.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=1, sampling_rate=100, frequency=[3, 6], noise = 0.2)
    >>>
    >>> k_max, info = nk.complexity_k(signal)

    Reference
    ----------
    - Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory.
    Physica D: Nonlinear Phenomena, 31(2), 277-283.

    - Vega, C. F., & Noel, J. (2015, June). Parameters analyzed of Higuchi's fractal dimension for EEG brain signals.
    In 2015 Signal Processing Symposium (SPSympo) (pp. 1-5). IEEE. https://ieeexplore.ieee.org/document/7168285
    """
    # Get the range of k-max values to be tested
    # ------------------------------------------
    if isinstance(k_max, str):  # e.g., "default"
        # upper limit for k value (max possible value)
        k_max = int(np.floor(len(signal) / 2))  # so that normalizing factor is positive

    if isinstance(k_max, int):
        k_range = np.arange(2, k_max + 1)
    elif isinstance(k_max, [list, np.ndarray, pd.Series]):
        k_range = np.array(k_max)
    else:
        warn(
            "k_max should be an int or a list of values of kmax to be tested.",
            category=NeuroKitWarning,
        )

    # Compute the slope for each kmax value
    # --------------------------------------
    slopes = np.zeros(len(k_range))
    intercepts = np.zeros(len(k_range))
    for i, k in enumerate(k_range):
        slopes[i], intercepts[i], _, _ = _complexity_k_slope(signal, k)

    # Find plateau (the saturation point of slope)
    # --------------------------------------------
    # Find slopes that are approaching the max
    k_indices = np.where(slopes >= 0.95 * np.max(slopes))[0]
    # drop indices for k <= 2 (which is the minimum value)
    k_indices = k_indices[k_range[k_indices] > 2]

    if len(k_indices) == 0:
        k_optimal = np.max(k_range)
        warn(
            f"The optimal kmax value detected is 2 or less. There may be no plateau in this case. You can inspect the plot by set `show=True`. We will return optimal k_max = {k_optimal} (the max).",
            category=NeuroKitWarning,
        )
        k_optimal = 2
    else:
        k_optimal = k_range[k_indices[0]]

    # Return optimal tau and info dict
    return k_optimal, {"Values": k_range, "Scores": slopes, "Intercepts": intercepts}


# =============================================================================
# Utilities
# =============================================================================


def _complexity_k_slope(signal, k):
    k_values = np.arange(1, k + 1)
    average_values = _complexity_k_average_values(signal, k_values)

    # Slope of best-fit line through points
    slope, intercept = -np.polyfit(np.log(k_values), np.log(average_values), 1)
    return slope, intercept, k_values, average_values


def _complexity_k_average_values(signal, k_values):
    """Step 3 of Vega & Noel (2015)"""
    n = len(signal)
    average_values = np.zeros(len(k_values))

    # Compute length of the curve, Lm(k)
    for i, k in enumerate(k_values):
        sets = np.zeros(k)
        for j, m in enumerate(range(1, k + 1)):
            n_max = int(np.floor((n - m) / k))
            normalization = (n - 1) / (n_max * k)
            Lm_k = np.sum(np.abs(np.diff(signal[m - 1 :: k], n=1))) * normalization
            sets[j] = Lm_k / k
        # Compute average value over k sets of Lm(k)
        L_k = np.sum(sets) / k
        average_values[i] = L_k
    return average_values

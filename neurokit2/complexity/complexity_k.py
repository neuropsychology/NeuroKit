from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning


def complexity_k(signal, k_max="default", show=False):
    """
    The optimal kmax is computed based on the point at which HFD values plateau for a range of kmax values.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series, np.ndarray, pd.DataFrame]
        The signal (i.e., a time series) in the form of a vector of values or in
        the form of an n-dimensional array (with a shape of len(channels) x len(samples))
        or dataframe.
    k_max : Union[int, str, list], optional
        Maximum number of interval times (should be greater than or equal to 3) to be tested.
    show : bool
        Visualise the slope of the curve for the selected kmax value.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=1, frequency=[3, 6], noise = 0.2)

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

    # Get the slope
    # -------------
    slope_values = np.zeros(len(k_range))

    average_values = _complexity_k_average_values(signal, k_range)


# =============================================================================
# Utilities
# =============================================================================


def _complexity_k_average_values(signal, k_range):
    average_values = np.zeros(len(k_range))
    # Compute length of the curve, Lm(k)
    for i, k in enumerate(range(1, len(k_range) + 1)):
        sets = []
        for m in range(1, k + 1):
            n_max = int(np.floor((len(signal) - m) / k))
            normalization = (len(signal) - 1) / (n_max * k)
            Lm_k = np.sum(np.abs(np.diff(signal[m - 1 :: k], n=1))) * normalization
            Lm_k = Lm_k / k
            sets.append(Lm_k)
        # Compute average value over k sets of Lm(k)
        L_k = np.sum(sets) / k
        average_values[i] = L_k
    return average_values

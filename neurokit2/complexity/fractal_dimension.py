# -*- coding: utf-8 -*-
import numpy as np


def fractal_dimension(signal, method="higuchi", kmax=5):
    """
    Computes fractal dimension according to Higuchi's Fractal Dimension (HFD) or
    Kat's Fractal Dimension (KFD).
    
    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        The algorithm used to compute frctal dimension, HFD (by calling "hfd" or "higuchi") or
        KFD (by calling "kfd" or "kat'z").
    
    Returns
    ----------
    float
        The fractal dimension.

    References
    ----------
    - Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory.
    Physica D: Nonlinear Phenomena, 31(2), 277-283.

    """
    if method in ["higuchi", "hfd"]:
        dimension = _fractal_dimension_higuchi(signal, kmax)
    
    return dimension
        

def _fractal_dimension_higuchi(signal, kmax=5):
    """Computes Higuchi's Fractal Dimension (HFD) from the time series.
    Value should fall between 1 and 2.
    
    For more information about k parameter selection, see https://ieeexplore.ieee.org/document/7168285

    Parameters
    ----------
    x : list or np.array
        One dimensional time series.
    kmax : int
        Maximum number of interval times (should be greater than or equal to 2).

    Returns
    -------
    slope : float
        Higuchi fractal dimension.
    """
    N = signal.size
    average_values = []
    # Compute length of the curve, Lm(k)
    for k in range(1, kmax + 1):
        sets = []
        for m in range(1, k + 1):
            n_max = int(np.floor((N - m) / k))
            normalization = (N - 1) / (n_max * (k ** 2))
            Lm_k = np.sum(np.abs(np.diff(signal[m-1::k], n=1))) * normalization
            sets.append(Lm_k)
        # Compute average value over k sets of Lm(k)
        L_k = np.sum(sets) / k
        average_values.append(L_k)

    # Slope of best-fit line through points
    k_values = np.arange(1, kmax + 1)
    slope, _ = - np.polyfit(np.log2(k_values), np.log2(average_values), 1)

    return slope

# -*- coding: utf-8 -*-
import numpy as np


def fractal_dimension(signal, method="higuchi", kmax=5):
    """
    Computes fractal dimension according to Higuchi's Fractal Dimension (HFD) or
    Katz's Fractal Dimension (KFD).
    
    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        The algorithm used to compute frctal dimension, HFD (by calling "hfd" or "higuchi") or
        KFD (by calling "kfd" or "katz").
    
    Returns
    ----------
    float
        The fractal dimension.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5, noise=10)
    >>>
    >>> hfd = nk.fractal_dimension(signal, method="higuchi")
    >>> hfd #doctest: +SKIP
    >>> kfd = nk.fractal_dimension(signal, method="katz")
    >>> kfd #doctest: +SKIP

    References
    ----------
    - Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory.
    Physica D: Nonlinear Phenomena, 31(2), 277-283.

    - Katz, M. J. (1988). Fractals and the analysis of waveforms.
    Computers in Biology and Medicine, 18(3), 145–156. doi:10.1016/0010-4825(88)90041-8.
    """

    if method in ["higuchi", "hfd"]:
        dimension = _fractal_dimension_higuchi(signal, kmax)
    elif method in ["katz", "katz's", "kfd"]:
        dimension = _fractal_dimension_katz(signal)

    return dimension
        

def _fractal_dimension_higuchi(signal, kmax=5):
    """Computes Higuchi's Fractal Dimension (HFD) from the time series.
    Value should fall between 1 and 2.
    
    For more information about k parameter selection, see https://ieeexplore.ieee.org/document/7168285

    Parameters
    ----------
    signal : list or np.array
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


def _fractal_dimension_katz(signal):

    """Computes Katz's Fractal Dimension (HFD), based on eucliean distances between
    successive points in the signal which are summed and averaged,
    and the maximum distance between the starting and any other point in the sample.

    Here, fractal dimensions range from 1.0 for straight lines, through
    approximately 1.15 for random-walk waveforms, to approaching 1.5 for the most
    convoluted waveforms.

    Parameters
    ----------
    signal : list or np.array
        One dimensional time series.

    Returns
    -------
    kfd : float
        Katz's fractal dimension.
    """
    # Define total length of curve
    dists = np.abs(np.diff(signal))
    length = np.sum(dists)

    # Average distance between successive points
    a = np.mean(dists)

    # Compute farthest distance between starting point and any other point
    d = np.max(np.abs(signal - signal[0]))

    kfd = np.log10(length/a) / (np.log10(length/a) + np.log10(d/length))

    return kfd

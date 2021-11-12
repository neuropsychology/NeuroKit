# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def fractal_katz(signal):

    """Computes Katz's Fractal Dimension (KFD), based on euclidean distances between
    successive points in the signal which are summed and averaged,
    and the maximum distance between the starting and any other point in the sample.

    Here, fractal dimensions range from 1.0 for straight lines, through
    approximately 1.15 for random-walk waveforms, to approaching 1.5 for the most
    convoluted waveforms.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    Returns
    -------
    kfd : float
        Katz's fractal dimension of the single time series.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute Katz's fractal dimension.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5, noise=10)
    >>>
    >>> kfd, parameters = nk.fractal_katz(signal)
    >>> kfd #doctest: +SKIP

    References
    ----------
    - Katz, M. J. (1988). Fractals and the analysis of waveforms.
    Computers in Biology and Medicine, 18(3), 145–156. doi:10.1016/0010-4825(88)90041-8.

    """

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # prepare parameters
    info = {}

    # if one signal time series
    out = _fractal_katz(signal)

    return out, info


def _fractal_katz(signal):

    # Define total length of curve
    dists = np.abs(np.diff(signal))
    length = np.sum(dists)

    # Average distance between successive points
    a = np.mean(dists)

    # Compute farthest distance between starting point and any other point
    d = np.max(np.abs(signal - signal[0]))

    kfd = np.log10(length/a) / (np.log10(d/a))

    return kfd

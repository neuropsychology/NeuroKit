# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


from .utils import _get_r
from .entropy_sample import entropy_sample


def entropy_multiscale(signal, dimension=2, r="default", scale="default"):
    """Compute the multiscale entropy (MSE).


    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically 2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    r : float
        Tolerance (i.e., filtering level - max absolute difference between segments). If 'default', will be set to 0.2 times the standard deviation of the signal.
    scale : str, int or list
        A list of scale factors of coarse graining. If 'default', will use ``range(len(signal) / (dimension + 10))`` (see discussion `here <https://github.com/neuropsychology/NeuroKit/issues/75#issuecomment-583884426>`_). If 'max', will use all scales until the length of the signal. If an integer, will create a range until the specified int.



    Returns
    ----------
    float
        The point-estimate of multiscale entropy (MSE) as a float value corresponding to the area under the MSE values curve.

    See Also
    --------
    entropy_shannon, entropy_approximate, entropy_sample, entropy_fuzzy

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> nk.entropy_multiscale(signal)
    38.26359811291708


    References
    -----------
    - `pyEntropy` <https://github.com/nikdon/pyEntropy>`_
    - Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy
        and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
    - Costa, M., Goldberger, A. L., & Peng, C. K. (2005). Multiscale entropy analysis of biological signals.
        Physical review E, 71(2), 021906.
    - Gow, B. J., Peng, C. K., Wayne, P. M., & Ahn, A. C. (2015). Multiscale entropy analysis of center-of-pressure
        dynamics in human postural control: methodological considerations. Entropy, 17(12), 7926-7947.
    - Norris, P. R., Anderson, S. M., Jenkins, J. M., Williams, A. E., & Morris Jr, J. A. (2008).
        Heart rate multiscale entropy at three hours predicts hospital mortality in 3,154 trauma patients. Shock, 30(1), 17-22.
    """
    r = _get_r(signal, r=r)

    # Select scale
    if scale is None or scale == "max":
        scale = range(len(signal))  # Set to max
    elif scale == "default":
        scale = range(int(len(signal) / (dimension + 10)))  # See https://github.com/neuropsychology/NeuroKit/issues/75#issuecomment-583884426
    elif isinstance(scale, int):
        scale = range(scale)

    # Initalize mse vector
    mse = np.zeros(len(scale))
    for i in scale:
        temp = _coarsegrained(signal, i+1)
        if len(temp) >= 4:
            mse[i] = entropy_sample(temp, delay=1, dimension=dimension, r=r)

    # Remove inf
    mse = mse[mse != np.inf]
    mse = mse[mse != -np.inf]

    # Area under the curve
    mse = np.trapz(mse)
    return mse




# =============================================================================
# Internal
# =============================================================================
#def _coarsegrained_composite(signal, scale):
#    """
#    """
#    n = len(signal)
#    b = n // scale
#    x = np.reshape(signal[0:b*scale], (b, scale))
#    coarsed = np.mean(x, axis=1)
#    return coarsed


def _coarsegrained(signal, scale):
    """Extract coarse-grained time series.

    The coarse-grained time series for a scale factor Tau are obtained by
    calculating the arithmetic mean of Tau neighboring values without overlapping.

    To obtain the coarse-grained time series at a scale factor of Tau ,the original
    time series is divided into non-overlapping windows of length Tau and the
    data points inside each window are averaged.

    This coarse-graining procedure is similar to moving averaging and the decimation of the original time series.
    The decimation procedure shortens the length of the coarse-grained time series by a factor of Tau.

    This is an efficient version of ``pd.Series(signal).rolling(window=scale).mean().iloc[0::].values[scale-1::scale]``.
    >>> import neurokit2 as nk
    >>> signal = nk.signal_simulate()
    >>> scale = 2
    >>> cs = _coarsegrained(signal, scale)
    """
    n = len(signal)
    b = n // scale
    x = np.reshape(signal[0:b*scale], (b, scale))
    coarsed = np.mean(x, axis=1)
    return coarsed

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


from .utils_get_r import _get_r
from .entropy_sample import entropy_sample


def entropy_multiscale(signal, order=2, r="default"):
    """Compute the multiscale entropy (MSE).


    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    order : int
        The embedding dimension (often denoted as 'm'), i.e., the length of compared run of data. Typically 1, 2 or 3.
    r : float
        Tolerance (i.e., filtering level - max absolute difference between segments). If 'default', will be set to 0.2 times the standard deviation of the signal.


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
    53.64066922587802


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
    max_scale = len(signal)  # Set to max

    # Initalize mse vector
    mse = np.zeros(max_scale)
    for i in range(max_scale):
        temp = _entropy_multiscale_granularizesignal(signal, i+1)
        if len(temp) >= 4:
            mse[i] = entropy_sample(temp, order, r)

    # Remove inf
    mse = mse[mse != np.inf]
    mse = mse[mse != -np.inf]

    # Area under the curve
    mse = np.trapz(mse)
    return mse




# =============================================================================
# Internal
# =============================================================================

def _entropy_multiscale_granularizesignal(signal, scale):
    n = len(signal)
    b = int(np.fix(n / scale))
    temp = np.reshape(signal[0:b*scale], (b, scale))
    cts = np.mean(temp, axis=1)
    return cts

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .utils import _get_r, _get_coarsegrained, _get_scale
from .entropy_sample import entropy_sample


def entropy_multiscale(signal, scale="default", dimension=2, r="default", show=False, **kwargs):
    """Compute the multiscale entropy (MSE).


    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    scale : str, int or list
        A list of scale factors used for coarse graining the time series. If 'default', will use ``range(len(signal) / (dimension + 10))`` (see discussion `here <https://github.com/neuropsychology/NeuroKit/issues/75#issuecomment-583884426>`_). If 'max', will use all scales until half the length of the signal. If an integer, will create a range until the specified int.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically 2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    r : float
        Tolerance (i.e., filtering level - max absolute difference between segments). If 'default', will be set to 0.2 times the standard deviation of the signal.



    Returns
    ----------
    float
        The point-estimate of multiscale entropy (MSE) as a float value corresponding to the area under the MSE values curvee, which is essentially the sum of sample entropy values over the range of scale factors.

    See Also
    --------
    entropy_shannon, entropy_approximate, entropy_sample, entropy_fuzzy

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> nk.entropy_multiscale(signal, show=True)
    0.2326107791897362


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
    - Liu, Q., Wei, Q., Fan, S. Z., Lu, C. W., Lin, T. Y., Abbod, M. F., & Shieh, J. S. (2012). Adaptive computation of multiscale entropy and its application in EEG signals for monitoring depth of anesthesia during surgery. Entropy, 14(6), 978-992.
    """
    r = _get_r(signal, r=r)
    scale_factors = _get_scale(signal, scale=scale, dimension=dimension)

    # Initalize mse vector
    mse = np.full(len(scale_factors), np.nan)
    for tau in scale_factors:
        y = _get_coarsegrained(signal, tau)
        if len(y) >= 10 ** dimension:  # Compute only if enough values (Liu et al., 2012)
            mse[i] = entropy_sample(y, delay=1, dimension=dimension, r=r, **kwargs)

    if show is True:
        plt.plot(scale_factors, mse)

    # Remove inf, nan and 0
    mse = mse[~np.isnan(mse)]
    mse = mse[mse != np.inf]
    mse = mse[mse != -np.inf]

    # Area under the curve, essentially the sum, normalized by the number of values (so it's close to the mean)
    mse = np.trapz(mse) / len(mse)
    return mse




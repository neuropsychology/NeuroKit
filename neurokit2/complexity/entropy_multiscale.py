# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from .entropy_sample import entropy_sample
from .utils import _get_coarsegrained, _get_coarsegrained_rolling, _get_r, _get_scale, _phi, _phi_divide


def entropy_multiscale(
    signal, scale="default", dimension=2, r="default", composite=False, refined=False, fuzzy=False, show=False, **kwargs
):
    """Multiscale entropy (MSE) and its Composite (CMSE), Refined (RCMSE) or fuzzy version.

    Python implementations of the multiscale entropy (MSE), the composite multiscale entropy (CMSE),
    the refined composite multiscale entropy (RCMSE) or their fuzzy version (FuzzyMSE, FuzzyCMSE or
    FuzzyRCMSE).

    This function can be called either via ``entropy_multiscale()`` or ``complexity_mse()``. Moreover,
    variants can be directly accessed via ``complexity_cmse()``, `complexity_rcmse()``,
    ``complexity_fuzzymse()`` and ``complexity_fuzzyrcmse()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    scale : str or int or list
        A list of scale factors used for coarse graining the time series. If 'default', will use
        ``range(len(signal) / (dimension + 10))`` (see discussion
        `here <https://github.com/neuropsychology/NeuroKit/issues/75#issuecomment-583884426>`_).
        If 'max', will use all scales until half the length of the signal. If an integer, will create
        a range until the specified int.
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns
        an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    r : float
        Tolerance (i.e., filtering level - max absolute difference between segments). If 'default',
        will be set to 0.2 times the standard deviation of the signal (for dimension = 2).
    composite : bool
        Returns the composite multiscale entropy (CMSE), more accurate than MSE.
    refined : bool
        Returns the 'refined' composite MSE (RCMSE; Wu, 2014)
    fuzzy : bool
        Returns the fuzzy (composite) multiscale entropy (FuzzyMSE, FuzzyCMSE or FuzzyRCMSE).
    show : bool
        Show the entropy values for each scale factor.
    **kwargs
        Optional arguments.


    Returns
    ----------
    float
        The point-estimate of multiscale entropy (MSE) as a float value corresponding to the area under
        the MSE values curvee, which is essentially the sum of sample entropy values over the range of
        scale factors.

    See Also
    --------
    entropy_shannon, entropy_approximate, entropy_sample, entropy_fuzzy

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> entropy1 = nk.entropy_multiscale(signal, show=True)
    >>> entropy1 #doctest: +SKIP
    >>> entropy2 = nk.entropy_multiscale(signal, show=True, composite=True)
    >>> entropy2 #doctest: +SKIP
    >>> entropy3 = nk.entropy_multiscale(signal, show=True, refined=True)
    >>> entropy3 #doctest: +SKIP


    References
    -----------
    - `pyEntropy` <https://github.com/nikdon/pyEntropy>`_

    - Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate
      entropy and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology,
      278(6), H2039-H2049.

    - Costa, M., Goldberger, A. L., & Peng, C. K. (2005). Multiscale entropy analysis of biological
      signals. Physical review E, 71(2), 021906.

    - Gow, B. J., Peng, C. K., Wayne, P. M., & Ahn, A. C. (2015). Multiscale entropy analysis of
      center-of-pressure dynamics in human postural control: methodological considerations. Entropy,
      17(12), 7926-7947.

    - Norris, P. R., Anderson, S. M., Jenkins, J. M., Williams, A. E., & Morris Jr, J. A. (2008).
      Heart rate multiscale entropy at three hours predicts hospital mortality in 3,154 trauma patients.
      Shock, 30(1), 17-22.

    - Liu, Q., Wei, Q., Fan, S. Z., Lu, C. W., Lin, T. Y., Abbod, M. F., & Shieh, J. S. (2012). Adaptive
      computation of multiscale entropy and its application in EEG signals for monitoring depth of
      anesthesia during surgery. Entropy, 14(6), 978-992.

    """
    return _entropy_multiscale(
        signal,
        scale=scale,
        dimension=dimension,
        r=r,
        composite=composite,
        fuzzy=fuzzy,
        refined=refined,
        show=show,
        **kwargs
    )


# =============================================================================
# Internal
# =============================================================================
def _entropy_multiscale(
    signal, scale="default", dimension=2, r="default", composite=False, fuzzy=False, refined=False, show=False, **kwargs
):

    r = _get_r(signal, r=r, dimension=dimension)
    scale_factors = _get_scale(signal, scale=scale, dimension=dimension)

    # Initalize mse vector
    mse = np.full(len(scale_factors), np.nan)
    for i, tau in enumerate(scale_factors):

        # Regular MSE
        if refined is False and composite is False:
            mse[i] = _entropy_multiscale_mse(signal, tau, dimension, r, fuzzy, **kwargs)

        # Composite MSE
        elif refined is False and composite is True:
            mse[i] = _entropy_multiscale_cmse(signal, tau, dimension, r, fuzzy, **kwargs)

        # Refined Composite MSE
        else:
            mse[i] = _entropy_multiscale_rcmse(signal, tau, dimension, r, fuzzy, **kwargs)

    if show is True:
        plt.plot(scale_factors, mse)

    # Remove inf, nan and 0
    mse = mse[~np.isnan(mse)]
    mse = mse[mse != np.inf]
    mse = mse[mse != -np.inf]

    # The MSE index is quantified as the area under the curve (AUC),
    # which is like the sum normalized by the number of values. It's similar to the mean.
    return np.trapz(mse) / len(mse)


# =============================================================================
# Methods
# =============================================================================
def _entropy_multiscale_mse(signal, tau, dimension, r, fuzzy, **kwargs):
    y = _get_coarsegrained(signal, tau)
    if len(y) < 10 ** dimension:  # Compute only if enough values (Liu et al., 2012)
        return np.nan

    return entropy_sample(y, delay=1, dimension=dimension, r=r, fuzzy=fuzzy, **kwargs)


def _entropy_multiscale_cmse(signal, tau, dimension, r, fuzzy, **kwargs):
    y = _get_coarsegrained_rolling(signal, tau)
    if y.size < 10 ** dimension:  # Compute only if enough values (Liu et al., 2012)
        return np.nan

    mse_y = np.full(len(y), np.nan)
    for i in np.arange(len(y)):
        mse_y[i] = entropy_sample(y[i, :], delay=1, dimension=dimension, r=r, fuzzy=fuzzy, **kwargs)

    return np.mean(mse_y)


def _entropy_multiscale_rcmse(signal, tau, dimension, r, fuzzy, **kwargs):
    y = _get_coarsegrained_rolling(signal, tau)
    if y.size < 10 ** dimension:  # Compute only if enough values (Liu et al., 2012)
        return np.nan

    # Get phi for all kth coarse-grained time series
    phi_ = np.full([len(y), 2], np.nan)
    for i in np.arange(len(y)):
        phi_[i] = _phi(y[i, :], delay=1, dimension=dimension, r=r, fuzzy=fuzzy, approximate=False, **kwargs)

    # Average all phi of the same dimension, then divide, then log
    return _phi_divide([np.mean(phi_[:, 0]), np.mean(phi_[:, 1])])

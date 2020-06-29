# -*- coding: utf-8 -*-
import numpy as np

from .utils import _get_embedded, _get_r, _phi


def entropy_approximate(signal, delay=1, dimension=2, r="default", corrected=False, **kwargs):
    """Approximate entropy (ApEn)

    Python implementations of the approximate entropy (ApEn) and its corrected version (cApEn).
    Approximate entropy is a technique used to quantify the amount of regularity and the unpredictability
    of fluctuations over time-series data. The advantages of ApEn include lower computational demand
    (ApEn can be designed to work for small data samples (< 50 data points) and can be applied in real
    time) and less sensitive to noise. However, ApEn is heavily dependent on the record length and lacks
    relative consistency.

    This function can be called either via ``entropy_approximate()`` or ``complexity_apen()``, and the
    corrected version via ``complexity_capen()``.


    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or
        to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns
        an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    r : float
        Tolerance (similarity threshold). It corresponds to the filtering level - max absolute difference
        between segments. If 'default', will be set to 0.2 times the standard deviation of the signal
        (for dimension = 2).
    corrected : bool
        If true, will compute corrected ApEn (cApEn), see Porta (2007).
    **kwargs
        Other arguments.

    See Also
    --------
    entropy_shannon, entropy_sample, entropy_fuzzy

    Returns
    ----------
    float
        The approximate entropy as float value.


    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> entropy1 = nk.entropy_approximate(signal)
    >>> entropy1 #doctest: +SKIP
    >>> entropy2 = nk.entropy_approximate(signal, corrected=True)
    >>> entropy2 #doctest: +SKIP


    References
    -----------
    - `EntroPy` <https://github.com/raphaelvallat/entropy>`_

    - Sabeti, M., Katebi, S., & Boostani, R. (2009). Entropy and complexity measures for EEG signal
      classification of schizophrenic and control participants. Artificial intelligence in medicine,
      47(3), 263-274.

    - Shi, B., Zhang, Y., Yuan, C., Wang, S., & Li, P. (2017). Entropy analysis of short-term heartbeat
      interval time series during regular walking. Entropy, 19(10), 568.

    """
    r = _get_r(signal, r=r)

    if corrected is False:
        # Get phi
        phi = _phi(signal, delay=delay, dimension=dimension, r=r, approximate=True, **kwargs)

        apen = np.abs(np.subtract(phi[0], phi[1]))

    if corrected is True:

        __, count1 = _get_embedded(
            signal, delay=delay, dimension=dimension, r=r, distance="chebyshev", approximate=True, **kwargs
        )
        __, count2 = _get_embedded(
            signal, delay=delay, dimension=dimension + 1, r=r, distance="chebyshev", approximate=True, **kwargs
        )

        # Limit the number of vectors to N - (dimension + 1) * delay
        upper_limit = len(signal) - (dimension + 1) * delay

        # Correction to replace the ratio of count1 and count2 when either is equal to 1
        # As when count = 1, only the vector itself is within r distance
        correction = 1 / upper_limit

        vector_similarity = np.full(upper_limit, np.nan)

        for i in np.arange(upper_limit):
            if count1.astype(int)[i] != 1 and count2.astype(int)[i] != 1:
                vector_similarity[i] = np.log(count2[i] / count1[i])
            else:
                vector_similarity[i] = np.log(correction)

        apen = -np.mean(vector_similarity)

    return apen

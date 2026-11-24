import numpy as np
import pandas as pd

from .optim_complexity_tolerance import _entropy_apen, complexity_tolerance
from .utils_entropy import _get_count


def entropy_approximate(signal, delay=1, dimension=2, tolerance="sd", corrected=False, **kwargs):
    """**Approximate entropy (ApEn) and its corrected version (cApEn)**

    Approximate entropy is a technique used to quantify the amount of regularity and the
    unpredictability of fluctuations over time-series data. The advantages of ApEn include lower
    computational demand (ApEn can be designed to work for small data samples (< 50 data points)
    and can be applied in real time) and less sensitive to noise. However, ApEn is heavily
    dependent on the record length and lacks relative consistency.

    This function can be called either via ``entropy_approximate()`` or ``complexity_apen()``, and
    the corrected version via ``complexity_capen()``.

    ApEn is defined (Pincus, 1991) as :math:`\\Phi^m(r) - \\Phi^{m+1}(r)`, where
    :math:`\\Phi^m(r)` is the average of the logarithm of the proportion of embedded vectors
    (self-matches included) within a distance *r* of each vector of dimension *m*. The value can
    be slightly negative when few vectors have neighbours other than themselves (e.g., small
    tolerance, high dimension or short signal), which reflects the self-match bias that motivated
    :func:`sample entropy <entropy_sample>`.


    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    tolerance : float
        Tolerance (often denoted as *r*), distance to consider two data points as similar. If
        ``"sd"`` (default), will be set to :math:`0.2 * SD_{signal}`. See
        :func:`complexity_tolerance` to estimate the optimal value for this parameter.
    corrected : bool
        If true, will compute corrected ApEn (cApEn), see Porta (2007).
    **kwargs
        Other arguments.

    See Also
    --------
    entropy_shannon, entropy_sample, entropy_fuzzy

    Returns
    ----------
    apen : float
        The approximate entropy of the single time series.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute approximate entropy.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=5)
      apen, parameters = nk.entropy_approximate(signal)
      apen

      capen, parameters = nk.entropy_approximate(signal, corrected=True)
      capen


    References
    -----------
    * Pincus, S. M. (1991). Approximate entropy as a measure of system complexity. Proceedings of
      the National Academy of Sciences, 88(6), 2297-2301.
    * Sabeti, M., Katebi, S., & Boostani, R. (2009). Entropy and complexity measures for EEG signal
      classification of schizophrenic and control participants. Artificial intelligence in medicine,
      47(3), 263-274.
    * Shi, B., Zhang, Y., Yuan, C., Wang, S., & Li, P. (2017). Entropy analysis of short-term
      heartbeat interval time series during regular walking. Entropy, 19(10), 568.

    """

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    # Store parameters
    info = {
        "Dimension": dimension,
        "Delay": delay,
        "Tolerance": complexity_tolerance(
            signal,
            method=tolerance,
            dimension=dimension,
            show=False,
        )[0],
        "Corrected": corrected,
    }

    # Compute index
    if corrected is False:
        # ApEn is implemented in 'utils_entropy.py' to avoid circular imports
        # as one of the method for optimizing tolerance relies on ApEn
        out, _ = _entropy_apen(signal, delay, dimension, info["Tolerance"], **kwargs)
    else:
        out = _entropy_capen(signal, delay, dimension, info["Tolerance"], **kwargs)

    return out, info


# =============================================================================
# Utils
# =============================================================================


def _entropy_capen(signal, delay, dimension, tolerance, **kwargs):
    __, count1, _ = _get_count(
        signal,
        delay=delay,
        dimension=dimension,
        tolerance=tolerance,
        approximate=True,
        **kwargs,
    )
    __, count2, _ = _get_count(
        signal,
        delay=delay,
        dimension=dimension + 1,
        tolerance=tolerance,
        approximate=True,
        **kwargs,
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

    return -np.mean(vector_similarity)

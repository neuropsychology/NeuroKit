import math
from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from .entropy_renyi import entropy_renyi
from .entropy_shannon import entropy_shannon
from .entropy_tsallis import entropy_tsallis
from .utils_complexity_ordinalpatterns import complexity_ordinalpatterns


def entropy_permutation(signal, delay=1, dimension=3, corrected=True, weighted=False, conditional=False, **kwargs):
    """**Permutation Entropy (PEn), its Weighted (WPEn) and Conditional (CPEn) forms**

    Permutation Entropy (PEn) is a robust measure of the complexity of a dynamic system by
    capturing the order relations between values of a time series and extracting a probability
    distribution of the ordinal patterns (see Henry and Judge, 2019). Using ordinal descriptors
    increases robustness to large artifacts occurring with low frequencies. PEn is applicable
    for regular, chaotic, noisy, or real-world time series and has been employed in the context of
    EEG, ECG, and stock market time series.

    Mathematically, it corresponds to the :func:`Shannon entropy <entropy_shannon>` after the
    signal has been made :func:`discrete <complexity_symbolize>` by analyzing the permutations in
    the time-embedded space. Each vector of *m* (``dimension``) values, spaced by ``delay``
    samples, is replaced by its ordinal pattern :math:`\\pi` (the permutation that sorts it, see
    :func:`complexity_ordinalpatterns`), one of the :math:`m!` possible ones, and PEn is the
    entropy of the relative frequencies of the patterns (Bandt & Pompe, 2002):

    .. math::

      PEn = -\\sum_{\\pi} p(\\pi) \\log_2 p(\\pi)

    However, the main shortcoming of traditional PEn is that no information besides the order
    structure is retained when extracting the ordinal patterns, which leads to several possible
    issues (Fadlallah et al., 2013). The **Weighted PEn** was developed to address these
    limitations by incorporating significant information (regarding the amplitude) from the
    original time series into the ordinal patterns: each vector is weighted by its variance, so
    that :math:`p(\\pi)` is the sum of the variances of the vectors of pattern :math:`\\pi`,
    divided by the sum of the variances of all the vectors.

    The **Conditional Permutation Entropy (CPEn)** is the *sorting entropy* of Bandt & Pompe
    (2002): the information gained by ranking the last value of a vector of *m + 1* values among
    the first *m*, once their order is known. The pattern :math:`\\pi_{m+1}` of each vector of
    *m + 1* values determines the pattern :math:`\\pi_m` of its first *m* values, and CPEn is the
    entropy of the former given the latter:

    .. math::

      CPEn = H(\\pi_{m+1}) - H(\\pi_m) = H(\\pi_{m+1} \\mid \\pi_m)

    Both entropies are computed from the same vectors of *m + 1* values (and, for the
    **Conditional Weighted PEn**, with the same weights, the variances of the *m + 1* values), so
    that CPEn lies between 0 and :math:`\\log(m+1)`, as the last value can take *m + 1* ranks.
    White noise reaches this maximum. The difference of the :func:`Rényi entropies <entropy_renyi>`
    has the same bounds (in nats). CPEn is not the *conditional entropy of ordinal patterns* of
    Unakafov & Keller (2014), which is the entropy of the pattern of *m* values that follows a given
    one (the pattern of the vector shifted by the delay), and which they show is at most the
    sorting entropy. For instance, for the logistic map with *r = 4*, whose Kolmogorov-Sinai
    entropy is :math:`\\ln(2) = 0.693` nats, the sorting entropy is about 0.89 nats for *m = 5*, and
    their conditional entropy about 0.66.

    This function can be called either via ``entropy_permutation()`` or ``complexity_pe()``.
    Moreover, variants can be directly accessed via ``complexity_wpe()`` and ``complexity_mspe()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*), at least 2, or 1 with
        ``conditional=True`` (CPEn is then the PEn of dimension 2). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    corrected : bool
        If ``True`` (default), PEn and WPEn are divided by their maximum, :math:`\\log(m!)` (the
        entropy of the uniform distribution of the :math:`m!` patterns, in the same unit, e.g.,
        :math:`\\log_2(m!)` for the default Shannon entropy in bits), so that they lie between 0
        and 1. CPEn is divided by its maximum, :math:`\\log(m+1)` (in the same unit), so that it
        also lies between 0 and 1, and is close to 1 for white noise. With :func:`entropy_tsallis`,
        whose difference is not maximal for white noise, CPEn is divided by its maximum,
        :math:`\\ln_q(m+1) \\max(1, (m!)^{1-q})`, where :math:`\\ln_q(m+1)` is the Tsallis
        entropy of the uniform distribution of *m + 1* states, so that it is below 1 for white noise
        if *q > 1* (e.g., 0.17 for *m = 3* and *q = 2*). Otherwise, return the permutation entropy
        in the unit of the entropy function (bits by default).
    weighted : bool
        If True, compute the weighted permutation entropy (WPE).
    conditional : bool
        If True, compute the conditional permutation entropy (CPEn).
    **kwargs
        Optional arguments, such as ``algorithm``, the function used to compute the entropy of the
        distribution of the patterns (:func:`entropy_shannon` by default, :func:`entropy_renyi`, or
        :func:`entropy_tsallis`), and its arguments (e.g., the logarithmic ``base`` of
        :func:`entropy_shannon`, the ``alpha`` of :func:`entropy_renyi` or the ``q`` of
        :func:`entropy_tsallis`).

    Returns
    -------
    PEn : float
        Permutation Entropy. It is NaN (with a warning) if the signal contains missing values or,
        for WPEn, if all the vectors are constant.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    complexity_ordinalpatterns, entropy_shannon, entropy_multiscale

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, sampling_rate=100, frequency=[5, 6], noise=0.5)

      # Permutation Entropy (uncorrected)
      pen, info = nk.entropy_permutation(signal, corrected=False)
      pen

      # Weighted Permutation Entropy (WPEn)
      wpen, info = nk.entropy_permutation(signal, weighted=True)
      wpen

      # Conditional Permutation Entropy (CPEn)
      cpen, info = nk.entropy_permutation(signal, conditional=True)
      cpen

      # Conditional Weighted Permutation Entropy (CWPEn)
      cwpen, info = nk.entropy_permutation(signal, weighted=True, conditional=True)
      cwpen

      # Conditional Renyi Permutation Entropy (CRPEn)
      crpen, info = nk.entropy_permutation(signal, conditional=True, algorithm=nk.entropy_renyi, alpha=2)
      crpen

    References
    ----------
    * Henry, M., & Judge, G. (2019). Permutation entropy and information recovery in nonlinear
      dynamic economic time series. Econometrics, 7(1), 10.
    * Fadlallah, B., Chen, B., Keil, A., & Principe, J. (2013). Weighted-permutation entropy: A
      complexity measure for time series incorporating amplitude information. Physical Review E, 87
      (2), 022911.
    * Zanin, M., Zunino, L., Rosso, O. A., & Papo, D. (2012). Permutation entropy and its main
      biomedical and econophysics applications: a review. Entropy, 14(8), 1553-1577.
    * Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time
      series. Physical review letters, 88(17), 174102.
    * Unakafov, A. M., & Keller, K. (2014). Conditional entropy of ordinal patterns. Physica D:
      Nonlinear Phenomena, 269, 94-102.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    if dimension < (1 if conditional else 2):
        raise ValueError(
            "The dimension of Permutation Entropy should be at least 2 (there is only one pattern of 1 value),"
            " or 1 with conditional=True."
        )

    info = {
        "Corrected": corrected,
        "Weighted": weighted,
        "Conditional": conditional,
        "Dimension": dimension,
        "Delay": delay,
    }

    signal = np.asarray(signal, dtype=float)
    if np.any(np.isnan(signal)):
        warn(
            "The signal contains missing values, for which the ordinal patterns are not defined. Returning NaN.",
            category=NeuroKitWarning,
        )
        return np.nan, info

    pen = _entropy_permutation(signal, dimension=dimension, delay=delay, weighted=weighted, conditional=conditional, **kwargs)

    if corrected:
        pen = pen / _entropy_permutation_max(dimension, conditional=conditional, **kwargs)

    return pen, info


# =============================================================================
# Permutation Entropy
# =============================================================================
def _entropy_permutation(
    signal,
    dimension=3,
    delay=1,
    weighted=False,
    conditional=False,
    algorithm=entropy_shannon,
    sorting="quicksort",
    **kwargs,
):
    # CPEn uses the vectors of m + 1 values
    _, info = complexity_ordinalpatterns(
        signal,
        dimension=dimension + 1 if conditional else dimension,
        delay=delay,
        algorithm=sorting,
    )
    codes = np.reshape(info["Uniques"], -1)

    # Weighted permutation entropy ----------------------------------------------
    if weighted is True:
        weights = np.var(info["Embedded"], axis=1)
        if np.sum(weights) == 0:
            warn(
                "All the embedded vectors are constant (of variance 0), so the Weighted Permutation"
                " Entropy is not defined. Returning NaN.",
                category=NeuroKitWarning,
            )
            return np.nan
    else:
        weights = np.ones(len(codes))

    # Compute entropy algorithm ------------------------------------------------
    pe = _entropy_permutation_patterns(codes, weights, algorithm=algorithm, **kwargs)

    # Conditional permutation entropy -------------------------------------------
    if conditional is True:
        # Patterns of the first m values of the same vectors (i.e., of the vectors of m values, but
        # the last ones), with the same weights, so that the difference is a conditional entropy
        _, info_m = complexity_ordinalpatterns(signal, dimension=dimension, delay=delay, algorithm=sorting)
        first = np.reshape(info_m["Uniques"], -1)[: len(codes)]
        pe = pe - _entropy_permutation_patterns(first, weights, algorithm=algorithm, **kwargs)

    return pe


def _entropy_permutation_patterns(codes, weights, algorithm=entropy_shannon, **kwargs):
    # (Weighted) relative frequencies of the patterns: the sum of the weights of their vectors
    freq = np.bincount(codes, weights=weights)
    return algorithm(freq=freq / np.sum(freq), **kwargs)[0]


def _entropy_permutation_max(dimension, conditional=False, algorithm=entropy_shannon, **kwargs):
    """Maximum of the permutation entropy, in the unit of the entropy function: the entropy of the
    uniform distribution of the m! ordinal patterns or, for CPEn, of the m + 1 ranks that the last
    value can take among the first m."""
    n = dimension + 1 if conditional else math.factorial(dimension)
    if algorithm is entropy_shannon:
        return np.log(n) / np.log(kwargs.get("base", 2))
    if algorithm is entropy_renyi:
        # The Rényi entropy (in nats) of the uniform distribution, whatever alpha
        return np.log(n)
    h_max = algorithm(freq=np.ones(n) / n, **kwargs)[0]
    if conditional and algorithm is entropy_tsallis:
        # The difference is the sum of the conditional entropies, each at most ln_q(m + 1), weighted
        # by p(pattern)^q, whose sum is at most 1 for q > 1 and (m!)^(1 - q) for q < 1
        h_max = h_max * max(1, math.factorial(dimension) ** (1 - kwargs.get("q", 1)))
    return h_max

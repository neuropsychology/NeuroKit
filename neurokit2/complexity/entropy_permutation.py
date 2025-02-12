import math

import numpy as np
import pandas as pd

from .entropy_shannon import entropy_shannon
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
    the time-embedded space.

    However, the main shortcoming of traditional PEn is that no information besides the order
    structure is retained when extracting the ordinal patterns, which leads to several possible
    issues (Fadlallah et al., 2013). The **Weighted PEn** was developed to address these
    limitations by incorporating significant information (regarding the amplitude) from the
    original time series into the ordinal patterns.

    The **Conditional Permutation Entropy (CPEn)** was originally defined by Bandt & Pompe as
    *Sorting Entropy*, but recently gained in popularity as conditional through the work of
    Unakafov et al. (2014). It describes the average diversity of the ordinal patterns succeeding a
    given ordinal pattern (dimension+1 vs. dimension).

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
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    corrected : bool
        If ``True``, divide by log2(factorial(m)) to normalize the entropy between 0 and 1. Otherwise,
        return the permutation entropy in bit.
    weighted : bool
        If True, compute the weighted permutation entropy (WPE).
    **kwargs
        Optional arguments, such as a function to compute Entropy (:func:`nk.entropy_shannon`
        (default), :func:`nk.entropy_tsallis` or :func:`nk.entropy_reyni`).

    Returns
    -------
    PEn : float
        Permutation Entropy
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    complexity_ordinalpatterns, entropy_shannon, entropy_multiscale

    Examples
    ----------
    .. ipython:: python

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

    info = {"Corrected": corrected, "Weighted": weighted, "Dimension": dimension, "Delay": delay}

    pen = _entropy_permutation(
        signal,
        dimension=dimension,
        delay=delay,
        corrected=corrected,
        weighted=weighted,
        **kwargs,
    )

    if conditional is True:
        # Compute PEn at m+1
        pen_m1 = _entropy_permutation(
            signal,
            dimension=dimension + 1,
            delay=delay,
            corrected=corrected,
            weighted=weighted,
            **kwargs,
        )
        # Get difference
        pen = pen_m1 - pen

        if corrected:
            pen = pen / np.log2(math.factorial(dimension + 1))
    else:
        if corrected:
            pen = pen / np.log2(math.factorial(dimension))

    return pen, info


# =============================================================================
# Permutation Entropy
# =============================================================================
def _entropy_permutation(
    signal,
    dimension=3,
    delay=1,
    corrected=True,
    weighted=False,
    algorithm=entropy_shannon,
    sorting="quicksort",
    **kwargs
):
    patterns, info = complexity_ordinalpatterns(
        signal,
        dimension=dimension,
        delay=delay,
        algorithm=sorting,
    )

    # Weighted permutation entropy ----------------------------------------------
    if weighted is True:
        info["Weights"] = np.var(info["Embedded"], axis=1)

        # Weighted frequencies of all permutations
        freq = np.array(
            [info["Weights"][np.all(info["Permutations"] == patterns[i], axis=1)].sum() for i in range(len(patterns))]
        )
        # Normalize
        freq = freq / info["Weights"].sum()
    else:
        freq = info["Frequencies"]

    # Compute entropy algorithm ------------------------------------------------
    pe, _ = algorithm(freq=freq, **kwargs)

    return pe

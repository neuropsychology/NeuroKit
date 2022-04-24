import numpy as np
import pandas as pd

from .complexity_coarsegraining import _get_scales, complexity_coarsegraining
from .complexity_embedding import complexity_embedding


def entropy_permutation(signal, dimension=3, delay=1, corrected=True, weighted=False, **kwargs):
    """**Permutation Entropy (PEn) and Weighted Permutation Entropy (WPEn)**

    Permutation Entropy (PEn) is a robust measure of the complexity of a dynamic system by
    capturing the order relations between values of a time series and extracting a probability
    distribution of the ordinal patterns (see Henry and Judge, 2019). Using ordinal descriptors
    increases robustness to large artifacts occurring with low frequencies. PEn is applicable
    for regular, chaotic, noisy, or real-world time series and has been employed in the context of
    EEG, ECG, and stock market time series.

    However, the main shortcoming of traditional PEn is that no information besides the order
    structure is retained when extracting the ordinal patterns, which leads to several possible
    issues (Fadlallah et al., 2013). The **Weighted PEn** was developped to address these
    limitations by incorporating significant information (regarding the amplitude) from the
    original time series into the ordinal patterns.

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
        :func:`complexity_dimension()` to estimate the optimal value for this parameter.
    corrected : bool
        If True, divide by log2(factorial(m)) to normalize the entropy between 0 and 1. Otherwise,
        return the permutation entropy in bit.
    weighted : bool
        If True, compute the weighted permutation entropy (WPE).
    **kwargs
        Optional arguments (currently not used).

    Returns
    -------
    PEn : float
        Permutation Entropy
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_multiscale

    Examples
    ----------
    .. ipython:: python

      signal = nk.signal_simulate(duration=2, sampling_rate=100, frequency=[5, 6], noise=0.5)

      # Permutation Entropy (uncorrected)
      pe, info = nk.entropy_permutation(signal, corrected=False)
      pe

      # Weighted Permutation Entropy
      wpe, info = nk.entropy_permutation(signal, dimension=3, weighted=True)
      wpe

    References
    ----------
    * Fadlallah, B., Chen, B., Keil, A., & Principe, J. (2013). Weighted-permutation entropy: A
      complexity measure for time series incorporating amplitude information. Physical Review E, 87
      (2), 022911.
    * Zanin, M., Zunino, L., Rosso, O. A., & Papo, D. (2012). Permutation entropy and its main
      biomedical and econophysics applications: a review. Entropy, 14(8), 1553-1577.
    * Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time
      series. Physical review letters, 88(17), 174102.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    info = {"Corrected": corrected, "Weighted": weighted}

    # Time-delay embedding
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    # Transform embedded into permutations matrix (ordinal patterns)
    permutations = embedded.argsort(kind="quicksort")

    # Weighted permutation entropy ----------------------------------------------
    if weighted is True:
        info["Weights"] = np.var(embedded, axis=1)

        patterns, freq = np.unique(permutations, axis=0, return_counts=True)

        # Add the weights of all permutations
        pw = np.array(
            [
                info["Weights"][np.all(permutations == patterns[i], axis=1)].sum()
                for i in range(len(patterns))
            ]
        )
        # Normalize
        pw = pw / info["Weights"].sum()

        # Compute WPEn
        pe = -np.dot(pw, np.log2(pw))

    # Normal permutation entropy ------------------------------------------------
    else:
        # Calculate relative frequency of each permutation
        _, freq = np.unique(permutations, axis=0, return_counts=True)
        freq = freq / freq.sum()

        # Compute PEn
        pe = -np.multiply(freq, np.log2(freq)).sum()

    # Apply correction
    if corrected:
        pe /= np.log2(np.math.factorial(dimension))

    return pe, info

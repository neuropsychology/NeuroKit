import numpy as np
import pandas as pd

from .complexity_embedding import complexity_embedding
from .utils import _get_coarsegrained, _get_scale


def entropy_permutation(signal, dimension=3, delay=1, corrected=True, weighted=False, scale=None):
    """Permutation Entropy (PE), and its Weighted (WPE) and/or Multiscale Variants (MSPE).

    Permutation Entropy (PE) is a robust measure of the complexity of a dynamic system by
    capturing the order relations between values of a time series and extracting a probability
    distribution of the ordinal patterns (see Henry and Judge, 2019). Using ordinal descriptors is
    helpful as it adds immunity to large artifacts occurring with low frequencies. PE is applicable
    for regular, chaotic, noisy, or real-world time series and has been employed in the context of
    EEG, ECG, and stock market time series.

    However, the main shortcoming of traditional PE is that no information besides the order
    structure is retained when extracting the ordinal patterns, which leads to several possible
    issues (Fadlallah et al., 2013). The Weighted PE was developped to address these limitations by
    incorporating significant information from the time series when retrieving the ordinal patterns.

    This function can be called either via ``entropy_permutation()`` or ``complexity_pe()``.
    Moreover, variants can be directly accessed via ``complexity_wpe()`` and ``complexity_mspe()``.

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
    corrected : bool
        If True, divide by log2(factorial(m)) to normalize the entropy between 0 and 1. Otherwise,
        return the permutation entropy in bit.
    weighted : bool
        If True, compute the weighted permutation entropy (WPE).
    scale : Union[list, str, list]
        If not ``None``, compute multiscale permutation entropy (MSPE). Can be a list of scale factors,
        or ``"default"`` or ``"max"``. See ``entropy_multiscale()`` for details.

    Returns
    -------
    PE : float
        Permutation Entropy
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_multiscale

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, sampling_rate=100, frequency=[5, 6], noise=0.5)
    >>>
    >>> # Permutation Entropy
    >>> pe, info = nk.entropy_permutation(signal, dimension=3, delay=1, corrected=False)
    >>> pe #doctest: +SKIP
    >>>
    >>> # Multiscale Permutation Entropy
    >>> mspe, info = nk.entropy_permutation(signal, dimension=3, scale = "default")
    >>> mspe #doctest: +SKIP
    >>>
    >>> # Weighted Permutation Entropy
    >>> wpe, info = nk.entropy_permutation(signal, dimension=3, weighted=True)
    >>> wpe #doctest: +SKIP

    References
    ----------
    - https://github.com/nikdon/pyEntropy
    - Fadlallah, B., Chen, B., Keil, A., & Principe, J. (2013). Weighted-permutation entropy: A
    complexity measure for time series incorporating amplitude information. Physical Review E, 87(2)
    , 022911.
    - Zanin, M., Zunino, L., Rosso, O. A., & Papo, D. (2012). Permutation entropy and its main
    biomedical and econophysics applications: a review. Entropy, 14(8), 1553-1577.
    - Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time
    series. Physical review letters, 88(17), 174102.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    info = {"Corrected": corrected, "Weighted": weighted, "Scale": None}

    # Multiscale
    if scale is not None:
        info["Scale"] = _get_scale(signal, scale=scale, dimension=dimension)
        info["Values"] = np.full(len(info["Scale"]), np.nan)
        for i, tau in enumerate(info["Scale"]):
            y = _get_coarsegrained(signal, tau)
            info["Values"][i] = _entropy_permutation(
                y,
                delay=1,
                dimension=dimension,
                corrected=corrected,
                weighted=weighted,
            )
        # Remove inf, nan and 0
        vals = info["Values"].copy()[~np.isnan(info["Values"])]
        vals = vals[vals != np.inf]
        vals = vals[vals != -np.inf]

        # The index is quantified as the area under the curve (AUC),
        # which is like the sum normalized by the number of values. It's similar to the mean.
        pe = np.trapz(vals) / len(vals)

    # Regular
    else:
        pe = _entropy_permutation(
            signal,
            dimension=dimension,
            delay=delay,
            corrected=corrected,
            weighted=weighted,
        )
    return pe, info


# =============================================================================
# Internal
# =============================================================================
def _entropy_permutation(signal, dimension=3, delay=1, corrected=True, weighted=False):
    # Time-delay embedding
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)

    if weighted is True:
        weights = np.var(embedded, axis=1)

    # Sort the order of permutations
    embedded = embedded.argsort(kind="quicksort")

    # Weighted permutation entropy ----------------------------------------------
    if weighted is True:
        motifs, c = np.unique(embedded, return_counts=True, axis=0)
        pw = np.zeros(len(motifs))
        for i, j in zip(weights, embedded):
            idx = int(np.where((j == motifs).sum(1) == dimension)[0])
            pw[idx] += i

        pw /= weights.sum()
        pe = -np.dot(pw, np.log2(pw))

    # Normal permutation entropy ------------------------------------------------
    else:
        # Associate unique integer to each permutations
        multiplier = np.power(dimension, np.arange(dimension))
        values = (np.multiply(embedded, multiplier)).sum(1)

        # Return the counts
        _, c = np.unique(values, return_counts=True)
        p = c / c.sum()
        pe = -np.multiply(p, np.log2(p)).sum()

    if corrected:
        pe /= np.log2(np.math.factorial(dimension))
    return pe

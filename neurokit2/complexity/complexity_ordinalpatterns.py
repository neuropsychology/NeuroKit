import numpy as np

from .complexity_embedding import complexity_embedding


def complexity_ordinalpatterns(signal, dimension=3, delay=1, **kwargs):
    """**Find Ordinal Patterns for Permutation Procedures**

    The seminal work by Bandt and Pompe (2002) introduced a symbolization approach to obtain a
    sequence of ordinal patterns (permutations) from continuous data. It is used in
    :func:`permutation entropy <entropy_permutation>` and its different variants.

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

    Returns
    -------
    array
        Ordinal patterns.
    vector
        Frequencies of each ordinal pattern.
    dict
        A dictionary containing additional elements.


    Examples
    ----------
    Example given by Bandt and Pompe (2002):

    .. ipython:: python

      import neurokit2 as nk

      signal = [4, 7, 9, 10, 6, 11, 3]

      patterns, freqs, info = nk.complexity_ordinalpatterns(signal, delay=1, dimension=3)
      patterns
      freqs


    References
    ----------
    * Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time
      series. Physical review letters, 88(17), 174102.

    """
    # Time-delay embedding
    info = {"Embedded": complexity_embedding(signal, delay=delay, dimension=dimension)}

    # Transform embedded into permutations matrix
    info["Permutations"] = info["Embedded"].argsort(kind="quicksort")

    # Count and get unique patterns
    patterns, freq = np.unique(info["Permutations"], axis=0, return_counts=True)

    # Relative Frequency
    freq = freq / freq.sum()

    # all_symbols = np.array(list(map(np.array, list(itertools.permutations(np.arange(delay * dimension))))))
    return patterns, freq, info

import numpy as np

from .utils_complexity_embedding import complexity_embedding


def complexity_ordinalpatterns(signal, delay=1, dimension=3, algorithm="quicksort", **kwargs):
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
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    algorithm : str
        Can be ``"quicksort"`` (default) or ``"bubblesort"`` (used in Bubble Entropy).

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

      patterns, info = nk.complexity_ordinalpatterns(signal, delay=1, dimension=3)
      patterns
      info["Frequencies"]

    .. ipython:: python

      signal = [4, 7, 9, 10, 6, 5, 3, 6, 8, 9, 5, 1, 0]

      patterns, info = nk.complexity_ordinalpatterns(signal, algorithm="bubblesort")
      info["Frequencies"]


    References
    ----------
    * Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time
      series. Physical review letters, 88(17), 174102.
    * Manis, G., Aktaruzzaman, M. D., & Sassi, R. (2017). Bubble entropy: An entropy almost free of
      parameters. IEEE Transactions on Biomedical Engineering, 64(11), 2711-2718.

    """
    # Time-delay embedding
    info = {"Embedded": complexity_embedding(signal, delay=delay, dimension=dimension)}

    # Transform embedded into permutations matrix
    if algorithm == "bubblesort":
        info["Permutations"] = _bubblesort(info["Embedded"])
    else:
        info["Permutations"] = info["Embedded"].argsort(kind="quicksort")

    # Count and get unique patterns
    patterns, info["Uniques"], info["Frequencies"] = np.unique(
        info["Permutations"],
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    # Find all possible patterns (not needed for now)
    # all_symbols = np.array(list(map(np.array, list(itertools.permutations(np.arange(delay * dimension))))))

    # Relative Frequency
    info["Frequencies"] = info["Frequencies"] / info["Frequencies"].sum()

    return patterns, info


def _bubblesort(embedded):
    """
    Manis, G., Aktaruzzaman, M. D., & Sassi, R. (2017). Bubble entropy: An entropy almost free of
    parameters. IEEE Transactions on Biomedical Engineering, 64(11), 2711-2718.
    """
    n, n_dim = np.shape(embedded)
    swaps = np.zeros(n)
    for y in range(n):
        for t in range(n_dim - 1):
            for kk in range(n_dim - t - 1):
                if embedded[y, kk] > embedded[y, kk + 1]:
                    embedded[y, kk], embedded[y, kk + 1] = embedded[y, kk + 1], embedded[y, kk]
                    swaps[y] += 1
    return swaps

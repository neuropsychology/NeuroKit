import numpy as np
import pandas as pd
import scipy.stats


def entropy_shannon(signal, base=2):
    """Shannon entropy (SE or ShanEn)

    Python implementation of Shannon entropy (SE). Entropy is a measure of unpredictability of the state,
    or equivalently, of its average information content. Shannon entropy (SE) is one of the first and
    most basic measure of entropy and a foundational concept of information theory. Shannon's entropy
    quantifies the amount of information in a variable. Shannon attempted to extend Shannon entropy in
    what has become known as Differential Entropy (see ``entropy_differential()``).

    Because Shannon entropy was meant for symbolic sequences (discrete events such as ["A", "B", "B", "A"]), it does
    not do well with continuous signals. One option is to binarize (i.e., cut) the signal into a number of
    bins using ``pd.cut(signal, bins=100, labels=False)``.

    This function can be called either via ``entropy_shannon()`` or ``complexity_se()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    base: float
        The logarithmic base to use, defaults to 2. Note that ``scipy.stats.entropy``
        uses ``np.e`` as default (the natural logarithm).

    Returns
    --------
    shanen : float
        The Shannon entropy of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute Shannon entropy.

    See Also
    --------
    entropy_differential, entropy_cumulative_residual, entropy_approximate, entropy_sample, entropy_fuzzy

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5, noise=0.1)
    >>> shanen, info = nk.entropy_shannon(signal)
    >>> shanen #doctest: +SKIP

    References
    -----------
    - `pyEntropy` <https://github.com/nikdon/pyEntropy>`_

    - `EntroPy` <https://github.com/raphaelvallat/entropy>`_

    - `nolds` <https://github.com/CSchoel/nolds>`_

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Check if string ('ABBA'), and convert each character to list (['A', 'B', 'B', 'A'])
    if not isinstance(signal, str):
        signal = list(signal)

    shanen = scipy.stats.entropy(pd.Series(signal).value_counts(), base=base)

    return shanen, {"Base": base}

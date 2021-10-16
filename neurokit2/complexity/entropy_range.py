import numpy as np
import pandas as pd


def entropy_range(signal, dimension=3, delay=1, r="default", method="mSampEn"):
    """Range Entropy (RangeEn)

    Introduced by `Omidvarnia et al. (2018) <https://www.mdpi.com/1099-4300/20/12/962/htm>`_.


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

    Returns
    -------
    RangeEn : float
        Range Entropy

    References
    ----------
    - Omidvarnia, A., Mesbah, M., Pedersen, M., & Jackson, G. (2018). Range entropy: A bridge between
    signal complexity and self-similarity. Entropy, 20(12), 962.


    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, sampling_rate=100, frequency=[5, 6], noise=0.5)
    >>>
    >>> # Permutation Entropy
    >>> rangeen, info = nk.entropy_range(signal, dimension=3, delay=1, corrected=False)
    >>> rangeen

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    return "SOON."

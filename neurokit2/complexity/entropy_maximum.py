import numpy as np


def entropy_maximum(signal):
    """**Maximum Entropy (MaxEn)**

    Provides an upper bound for the entropy of a random variable, so that the empirical entropy
    (obtained for instance with :func:`entropy_shannon`) will lie in between 0 and max. entropy.

    It can be useful to normalize the empirical entropy by the maximum entropy (which is made by
    default in some algorithms).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    Returns
    --------
    maxen : float
        The maximum entropy of the signal.
    info : dict
        An empty dictionary returned for consistency with the other complexity functions.

    See Also
    --------
    entropy_shannon

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = [1, 1, 5, 5, 2, 8, 1]
      maxen, _ = nk.entropy_maximum(signal)
      maxen


    """
    return np.log2(len(np.unique(signal))), {}

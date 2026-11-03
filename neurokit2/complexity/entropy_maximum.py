import numpy as np

from .entropy_shannon import _entropy_freq


def entropy_maximum(signal):
    """**Maximum Entropy (MaxEn)**

    Provides an upper bound for the entropy of a random variable, so that the empirical entropy
    (obtained for instance with :func:`entropy_shannon`) will lie in between 0 and max. entropy.
    It is the entropy of a uniform distribution over the *k* distinct states (values) of the
    signal, :math:`\\log_2(k)`, in *bits*.

    It can be useful to normalize the empirical entropy by the maximum entropy (which is made by
    default in some algorithms).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series, str]
        The signal (i.e., a time series) in the form of a vector of values. A string (e.g.,
        ``"ABBA"``) is treated as a sequence of characters, as in :func:`entropy_shannon`.

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

      # The Shannon entropy lies in between 0 and the maximum entropy
      nk.entropy_shannon(signal)[0]

    """
    states, _ = _entropy_freq(signal)
    return np.log2(len(states)), {}

import numpy as np
import scipy.stats

from .entropy_shannon import _entropy_freq


def entropy_shannon_joint(x, y, base=2):
    """**Shannon's Joint Entropy**

    The joint entropy measures how much entropy is contained in a joint system of two random
    variables.

    Parameters
    ----------
    x : Union[list, np.array, pd.Series]
        A :func:`symbolic <complexity_symbolize>` sequence in the form of a vector of values.
    y : Union[list, np.array, pd.Series]
        Another symbolic sequence with the same values.
    base: float
        The logarithmic base to use, defaults to ``2``. Note that ``scipy.stats.entropy()``
        uses ``np.e`` as default (the natural logarithm).

    Returns
    --------
    float
        The Shannon joint entropy.
    dict
        A dictionary containing additional information regarding the parameters used
        to compute Shannon entropy.

    See Also
    --------
    entropy_shannon


    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      x = ["A", "A", "A", "B", "A", "B"]
      y = ["A", "B", "A", "A", "A", "A"]

      jen, _ = nk.entropy_shannon_joint(x, y)
      jen

    """
    # Get frequencies
    labels_x, freq_x = _entropy_freq(x)
    labels_y, freq_y = _entropy_freq(y)

    assert np.all(labels_y == labels_y), "The labels of x and y are not the same."

    return scipy.stats.entropy(freq_x, freq_y, base=base), {"Base": base}

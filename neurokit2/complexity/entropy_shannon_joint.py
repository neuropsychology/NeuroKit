import numpy as np
import pandas as pd
import scipy.stats


def entropy_shannon_joint(x, y, base=2):
    """**Shannon's Joint Entropy**

    The joint entropy measures how much entropy is contained in a joint system of two random
    variables. It is the :func:`Shannon entropy <entropy_shannon>` of the pairs of values
    :math:`(x_i, y_i)` observed at the same time (Cover & Thomas, 2006):

    .. math::

      H(X, Y) = -\\sum_{x \\in \\mathcal{X}} \\sum_{y \\in \\mathcal{Y}} p(x, y) \\log_b p(x, y)

    It is at least the largest of the entropies of *X* and *Y*, and at most their sum (reached when
    they are independent).

    Parameters
    ----------
    x : Union[list, np.array, pd.Series, str]
        A :func:`symbolic <complexity_symbolize>` sequence in the form of a vector of values (a
        string, e.g., ``"ABBA"``, is treated as a sequence of characters).
    y : Union[list, np.array, pd.Series, str]
        Another symbolic sequence of the same length (its values can differ from those of ``x``).
    base: float
        The logarithmic base to use, defaults to ``2``, giving a unit in *bits*. Note that
        ``scipy.stats.entropy()`` uses ``np.e`` as default (the natural logarithm).

    Returns
    --------
    float
        The Shannon joint entropy.
    dict
        A dictionary containing additional information regarding the parameters used
        to compute Shannon entropy.

    See Also
    --------
    entropy_shannon, mutual_information


    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      x = ["A", "A", "A", "B", "A", "B"]
      y = ["A", "B", "A", "A", "A", "A"]

      jen, _ = nk.entropy_shannon_joint(x, y)
      jen

      # Between the largest entropy and the sum of the two entropies
      nk.entropy_shannon(x)[0], nk.entropy_shannon(y)[0]

    References
    -----------
    * Cover, T. M., & Thomas, J. A. (2006). Elements of information theory (2nd ed.). John Wiley &
      Sons.

    """
    x = _entropy_shannon_joint_sequence(x)
    y = _entropy_shannon_joint_sequence(y)
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length (got {len(x)} and {len(y)}).")

    # Code each pair of states (x_i, y_i) as a single integer and count them
    _, x_codes = np.unique(x, return_inverse=True)
    y_states, y_codes = np.unique(y, return_inverse=True)
    _, freq = np.unique(x_codes.ravel() * len(y_states) + y_codes.ravel(), return_counts=True)

    return scipy.stats.entropy(freq, base=base), {"Base": base}


def _entropy_shannon_joint_sequence(x):
    if isinstance(x, (np.ndarray, pd.DataFrame)) and x.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    # Check if string ('ABBA'), and convert each character to list (['A', 'B', 'B', 'A'])
    if isinstance(x, str):
        x = list(x)
    return np.asarray(x)

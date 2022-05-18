import numpy as np
import pandas as pd


def check_type(x, what="str"):
    """**Check type of input**

    Creates a list of boolean values to check if the input is of the target type.

    Parameters
    ----------
    x : int, list, pd.DataFrame, pd.Series, np.ndarray
        Target of checking
    what : str
        Compare the dtype of target with what.

    Returns
    -------
    array
        An array of boolean values.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      nk.check_type([1, 2, 3, "hello"], what="str")

      nk.check_type(pd.DataFrame({"A": [3, 1, 2, 4, 6, np.nan],
                                  "B": [3, 1, 2, "hello", 6, 5]}), what="str")

    """

    if what == "str":
        out = is_string(x)
    return out


def is_string(x):
    if isinstance(x, list):
        out = [isinstance(member, str) for member in x]
    elif isinstance(x, pd.DataFrame):
        out = [member == 'object' for member in list(x.dtypes)]
    elif isinstance(x, pd.Series):
        out = [x.dtype == "object"]
    elif isinstance(x, np.ndarray):
        out = [x.dtype == "U1"]
    else:
        out = isinstance(x, str)
    return np.array(out)

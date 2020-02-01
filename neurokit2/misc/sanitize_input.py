# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def sanitize_input(x, what="vector", message="NeuroKit error: please provide a correct input."):
    """Make sure that the input is of the right shape.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> nk.sanitize_input(x=range(3))
    >>> nk.sanitize_input(x=[0, 1, 2])
    >>> nk.sanitize_input(x=np.array([0, 1, 2]))
    >>> nk.sanitize_input(x=0)
    >>> nk.sanitize_input(x=pd.Series([0, 1, 2]))
    >>> nk.sanitize_input(x=pd.DataFrame([0, 1, 2]))
    """
    if what == "vector":
        out = sanitize_input_vector(x, message)
    else:
        raise ValueError("NeuroKit error: sanitize_input(): 'what' should be "
                         "one of 'vector'.")

    return out





def sanitize_input_vector(x, message="NeuroKit error: we expect the user to provide a vector, i.e., a one-dimensional array (such as a list of values)."):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        out = x.values
    elif isinstance(x, (str, float, int)):
        out = np.array([x])
    else:
        out = np.array(x)


    if isinstance(out, np.ndarray):
        shape = out.shape
        if len(shape) == 1:
            pass
        elif len(shape) != 1 and len(shape) == 2 and shape[1] == 1:
            out = out[:, 0]
        else:
            raise ValueError(message + " Current input of shape " + str(shape))

    return out

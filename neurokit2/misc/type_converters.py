# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd






def as_vector(x):
    """Convert to vector.

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
    if isinstance(x, (pd.Series, pd.DataFrame)):
        out = x.values
    elif isinstance(x, (str, float, int, np.int, np.int8, np.int16, np.int32, np.int64)):
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
            raise ValueError("NeuroKit error: we expect the user to provide a "
                             "vector, i.e., a one-dimensional array (such as a "
                             "list of values). Current input of shape: " + str(shape))

    return out

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def signal_binarize(signal, threshold="auto"):
    """Binarize a continuous signal.

    Parameters
    ----------
    signal : array or list
        The signal channel.
    threshold : float
        The threshold value by which to select the events. If "auto", takes the value between the max and the min.

    Returns
    -------
    list
        A list or array depending on the type passed.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    >>> binary = nk.signal_binarize(signal)
    >>> pd.DataFrame({"Raw": signal, "Binary": binary}).plot()
    """
    # Sanity check
    if isinstance(signal, list):
        signal = np.array(signal)
        return_list = True
    else:
        signal = signal.copy()
        return_list = False

    if threshold == "auto":
        threshold = np.mean([np.max(signal), np.min(signal)])

    signal[signal > threshold] = 1
    signal[signal <= threshold] = 0

    if return_list:
        return(list(signal))
    else:
        return(signal)


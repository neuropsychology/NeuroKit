# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np





def _signal_binarize(signal, threshold="auto"):
    if threshold == "auto":
        threshold = np.mean([np.max(signal), np.min(signal)])

    binary = signal.copy()
    binary[signal > threshold] = 1
    binary[signal <= threshold] = 0
    return(binary)







def signal_binarize(signal, threshold="auto"):
    """Binarize a continuous signal.

    Convert a continuous signal into zeros and ones depending on a given threshold.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
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

    # Return appropriate type
    if isinstance(signal, list):
        binary = _signal_binarize(np.array(signal), threshold=threshold)
        signal = list(binary)
    elif isinstance(signal, pd.Series):
        binary = _signal_binarize(signal.values, threshold=threshold)
        signal[:] = binary
    else:
        signal = _signal_binarize(signal, threshold=threshold)

    return(signal)



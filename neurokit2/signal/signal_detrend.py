# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def signal_detrend(signal, order=1):
    """Polynomial detrending of signal.

    Function to do baseline (order = 0), linear (order = 1), or polynomial (order > 1) detrending of the signal (i.e., removing a general trend).

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    order : int
        The order of the polynomial. 0, 1 or > 1 for a baseline ('constant detrend', i.e., remove only the mean), linear (remove the linear trend) or polynomial detrending.

    Returns
    -------
    array
        Vector containing the detrended signal.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=10, num=1000))  # Low freq
    >>> signal += np.cos(np.linspace(start=0, stop=100, num=1000))  # High freq
    >>> signal += 3  # Add baseline
    >>>
    >>> pd.DataFrame({"Raw": signal,
                      "Baseline_Detrend": nk.signal_detrend(signal, order=0),
                      "Linear_Detrend": nk.signal_detrend(signal, order=1),
                      "Quadratic_Detrend": nk.signal_detrend(signal, order=2),
                      "Cubic_Detrend": nk.signal_detrend(signal, order=3),
                      "Crazy_Detrend": nk.signal_detrend(signal, order=150)}).plot()
    """
    x_axis = np.linspace(0, 100, num=len(signal))
    # Generating weights and model for polynomial function with a given degree
    model = np.poly1d(np.polyfit(x_axis, signal, order))
    trend = model(x_axis)
    # detrend
    detrended = np.array(signal) - trend
    return(detrended)

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import sklearn.linear_model


def signal_detrend(signal, method="linear", degree=1):
    """
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
    >>> detrended = signal_detrend(signal, degree=1)
    >>> pd.DataFrame({"Raw": signal,
                      "Baseline_Detrend": signal_detrend(signal, degree=0),
                      "Linear_Detrend": signal_detrend(signal, degree=1),
                      "Quadratic_Detrend": signal_detrend(signal, degree=2),
                      "Cubic_Detrend": signal_detrend(signal, degree=3)}).plot()
    """
    x_axis = np.linspace(0, 100, num=len(signal))
    # Generating weights and model for polynomial function with degree =2
    model = np.poly1d(np.polyfit(x_axis, signal, degree))
    trend = model(x_axis)
    # detrend
    detrended = np.array(signal) - trend
    return(detrended)

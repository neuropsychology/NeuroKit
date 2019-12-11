# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from scipy.signal import detrend

def signal_detrend(signal):
    """
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=10, num=1000)) # Low freq
    >>> signal += np.cos(np.linspace(start=0, stop=100, num=1000)) # High freq
    >>> detrended = signal_detrend(signal, highcut=10)
    >>> pd.DataFrame({"Raw": signal, "Filtered": filtered}).plot()
    """

    return(signal)

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .rsp_prepare import rsp_prepare
from .rsp_findpeaks import rsp_findpeaks
from .rsp_rate import rsp_rate


def rsp_process(rsp_signal, sampling_rate=1000):
    """RSP (respiration) signal processing.

    Examples
    ---------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=50, num=10000))
    >>> data, info = nk.rsp_process(signal, sampling_rate=1000)
    >>> nk.standardize(data).plot()
    """
    preprocessed = rsp_prepare(rsp_signal, sampling_rate=sampling_rate)

    # Extremas (peaks and troughs)
    peaks, info = rsp_findpeaks(preprocessed["RSP_Filtered"], sampling_rate=sampling_rate, outlier_threshold=0.3)

    rate = rsp_rate(peaks=info["RSP_Peaks"],
                    troughs=info["RSP_Troughs"],
                    desired_length=len(rsp_signal),
                    sampling_rate=sampling_rate)

    data = pd.concat([preprocessed, peaks, rate], axis=1)
    return(data, info)



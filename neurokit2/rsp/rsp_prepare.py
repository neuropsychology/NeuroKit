# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_detrend
from ..signal import signal_filter
from ..signal import signal_interpolate

from .rsp_findpeaks import rsp_findpeaks








def rsp_prepare(rsp, sampling_rate=1000):
    """Preprocessing of RSP (respiration) signal

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=40, num=20000))
    >>> data = nk.rsp_prepare(signal, sampling_rate=1000)
    >>> data.plot()
    """
    # Detrend and lowpass-filter the signal to be able to reliably detect
    # zero crossings in raw signal.
    rsp = signal_detrend(rsp, order=1)
    filtered_rsp = signal_filter(rsp, sampling_rate=sampling_rate, highcut=2, method="butterworth")

    # Prepare output
    data = pd.DataFrame({"RSP_Raw": rsp,
                         "RSP_Filtered": filtered_rsp})
    return(data)
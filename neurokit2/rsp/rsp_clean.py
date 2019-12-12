# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_detrend
from ..signal import signal_filter
from ..signal import signal_interpolate






def rsp_clean(rsp_signal, sampling_rate=1000):
    """Preprocessing of RSP (respiration) signal

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=40, num=20000))
    >>> data = nk.rsp_clean(signal, sampling_rate=1000)
    >>> data.plot()
    """
    # Detrend and lowpass-filter the signal to be able to reliably detect
    # zero crossings in raw signal.
    filtered_rsp = signal_detrend(rsp_signal, order=1)
    filtered_rsp = signal_filter(filtered_rsp, sampling_rate=sampling_rate, highcut=2, method="butterworth")

    # Prepare output
    data = pd.DataFrame({"RSP_Raw": rsp_signal,
                         "RSP_Filtered": filtered_rsp})
    return(data)
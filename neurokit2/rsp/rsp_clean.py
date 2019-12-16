# -*- coding: utf-8 -*-
import pandas as pd

from ..signal import signal_detrend
from ..signal import signal_filter


def rsp_clean(rsp_signal, sampling_rate=1000):
    """Preprocess a respiration (RSP) signal.

    This function applies linear detrending, followed by an IIR Butterworth
    lowpass filter.

    Parameters
    ----------
    rsp_signal : list, array or Series
        The raw respiration channel (as measured, for instance, by a
        respiration belt).
    sampling_rate : int
        The sampling frequency of rsp_signal (in Hz, i.e., samples/second).

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as the input signal containing the raw signal
        and the cleaned signal, accessible with the keys "RSP_Raw", and
        "RSP_Filtered" respectively.

    See Also
    --------
    rsp_findpeaks, rsp_rate, rsp_process, rsp_plot

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> rsp = np.cos(np.linspace(start=0, stop=40, num=20000))
    >>> signals = nk.rsp_clean(rsp, sampling_rate=1000)
    >>> signals.plot()
    """
    # Detrend and lowpass-filter the signal to be able to reliably detect
    # zero crossings in raw signal.
    filtered_rsp = signal_detrend(rsp_signal, order=1)
    filtered_rsp = signal_filter(filtered_rsp, sampling_rate=sampling_rate,
                                 highcut=2, method="butterworth")

    # Prepare output
    signals = pd.DataFrame({"RSP_Raw": rsp_signal,
                            "RSP_Filtered": filtered_rsp})
    return(signals)

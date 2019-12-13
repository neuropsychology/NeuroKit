# -*- coding: utf-8 -*-
import pandas as pd

from .rsp_clean import rsp_clean
from .rsp_findpeaks import rsp_findpeaks
from .rsp_rate import rsp_rate


def rsp_process(rsp_signal, sampling_rate=1000):
    """Process a respiration (RSP) signal.

    Convenience function that automatically processes a respiration signal.

    Parameters
    ----------
    rsp_signal : list, array or Series
        The raw respiration channel (as measured, for instance, by a
        respiration belt).
    sampling_rate : int, default 1000
        The sampling frequency of rsp_signal (in Hz, i.e., samples/second).

    Returns
    -------
    signals : DataFrame
        A DataFrame f same length as the input signal containing the following
        columns:
        - "RSP_Raw": the raw signal.
        - "RSP_Filtered": the cleaned signal.
        - "RSP_Peaks": the inhalation peaks marked as "1" in a list
            of zeros.
        - "RSP_Troughs": the exhalation troughs marked as "1" in a
            list of zeros.
        - "RSP_Rate": breathing rate interpolated between inhalation peaks.
        - "RSP_Period": the breathing period interpolated between inhalation peaks.
        - "RSP_Amplitude": the breathing amplitude interpolated between inhalation
            peaks.
    info : dict
        A dictionary containing additional information, in this case the samples
        at which inhalation peaks and exhalation troughs occur, accessible with
        the keys 'RSP_Peaks', and 'RSP_Troughs', respectively.

    See Also
    --------
    rsp_clean, rsp_findpeaks, rsp_rate, rsp_plot

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> rsp = np.cos(np.linspace(start=0, stop=50, num=10000))
    >>> signals, info = nk.rsp_process(rsp, sampling_rate=1000)
    >>> nk.signal_plot(nk.standardize(signals))
    """
    preprocessed = rsp_clean(rsp_signal, sampling_rate=sampling_rate)

    extrema_signal, info = rsp_findpeaks(preprocessed["RSP_Filtered"],
                                            sampling_rate=sampling_rate,
                                            outlier_threshold=0.3)

    rate = rsp_rate(extrema_signal, sampling_rate=sampling_rate)

    signals = pd.concat([preprocessed, extrema_signal, rate], axis=1)
    return(signals, info)

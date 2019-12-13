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
    summary : DataFrame
        A DataFrame containing:
        1. raw signal (key : "RSP_Raw").
        2. cleaned signal (key: "RSP_Filtered").
        3. inhalation peaks (key: "RSP_Peaks_Signal") marked as "1" in a list
            of zeros with the same length as rsp_signal.
        4. exhalation troughs (key: "RSP_Troughs_Signal") marked as "1" in a
            list of zeros with the same length as rsp_signal.
        5. rate (key : "RSP_Rate") interpolated between inhalation peaks over
            the length of rsp_signal.
        6. period (key : "RSP_Period") interpolated between inhalation peaks
            over the length of rsp_signal.
        7. amplitude (key : "RSP_Amplitude") interpolated between inhalation
            peaks over the length of rsp_signal.
    extrema: dict
        A dictionary containing the samples at which inhalation peaks and
        exhalation troughs occur, accessible with the keys "RSP_Peaks", and
        "RSP_Troughs" respectively.

    See Also
    --------
    rsp_plot

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=50, num=10000))
    >>> data, info = nk.rsp_process(signal, sampling_rate=1000)
    >>> nk.signal_plot(nk.standardize(data))
    """
    preprocessed = rsp_clean(rsp_signal, sampling_rate=sampling_rate)

    extrema_signal, extrema = rsp_findpeaks(preprocessed["RSP_Filtered"],
                                            sampling_rate=sampling_rate,
                                            outlier_threshold=0.3)

    rate = rsp_rate(peaks=extrema["RSP_Peaks"],
                    troughs=extrema["RSP_Troughs"],
                    return_amplitude=True,
                    desired_length=len(rsp_signal),
                    sampling_rate=sampling_rate)

    summary = pd.concat([preprocessed, extrema_signal, rate], axis=1)
    return(summary, extrema)

# -*- coding: utf-8 -*-
import pandas as pd

from .rsp_clean import rsp_clean
from .rsp_findpeaks import rsp_findpeaks
from .rsp_rate import rsp_rate


def rsp_process(rsp_signal, sampling_rate=1000, method="khodadad2018"):
    """Process a respiration (RSP) signal.

    Convenience function that automatically processes a respiration signal.

    Parameters
    ----------
    rsp_signal : list, array or Series
        The raw respiration channel (as measured, for instance, by a
        respiration belt).
    sampling_rate : int
        The sampling frequency of rsp_signal (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of 'khodadad2018' (default) or 'biosppy'.

    Returns
    -------
    signals : DataFrame
        A DataFrame f same length as the input signal containing the following
        columns:

        - *"RSP_Raw"*: the raw signal.
        - *"RSP_Clean"*: the cleaned signal.
        - *"RSP_Peaks"*: the inhalation peaks marked as "1" in a list of zeros.
        - *"RSP_Troughs"*: the exhalation troughs marked as "1" in a list of zeros.
        - *"RSP_Rate"*: breathing rate interpolated between inhalation peaks.
        - *"RSP_Amplitude"*: the breathing amplitude interpolated between inhalation peaks.
    info : dict
        A dictionary containing the samples at which inhalation peaks and
        exhalation troughs occur, accessible with the keys 'RSP_Peaks', and
        'RSP_Troughs', respectively.

    See Also
    --------
    rsp_clean, rsp_findpeaks, rsp_rate, rsp_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration=90, respiratory_rate=15)
    >>> signals, info = nk.rsp_process(rsp, sampling_rate=1000)
    >>> nk.rsp_plot(signals)
    """
    rsp_cleaned = rsp_clean(rsp_signal, sampling_rate=sampling_rate, method=method)

    extrema_signal, info = rsp_findpeaks(rsp_cleaned, method=method, outlier_threshold=0.3)

    rate = rsp_rate(extrema_signal, sampling_rate=sampling_rate, method=method)

    signals = pd.DataFrame({"RSP_Raw": rsp_signal,
                            "RSP_Clean": rsp_cleaned})
    signals = pd.concat([signals, extrema_signal, rate], axis=1)
    return signals, info

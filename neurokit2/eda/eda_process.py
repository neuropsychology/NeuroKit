# -*- coding: utf-8 -*-
import pandas as pd

from .eda_clean import eda_clean
from .eda_peaks import eda_peaks
from .eda_phasic import eda_phasic


def eda_process(eda_signal, sampling_rate=1000):
    """
    Process Electrodermal Activity (EDA).

    Convenience function that automatically processes electrodermal activity (EDA) signal.

    Parameters
    ----------
    eda_signal : list, array or Series
        The raw EDA signal.
    sampling_rate : int
        The sampling frequency of `rsp_signal` (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of "khodadad2018"
        (default) or "biosppy".

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as `eda_signal` containing the following
        columns:

        - *"EDA_Raw"*: the raw signal.
        - *"EDA_Clean"*: the cleaned signal.
        - *"EDA_Tonic"*: the tonic component of the signal, or the Tonic Skin Conductance Level (SCL).
        - *"EDA_Phasic"*: the phasic component of the signal, or the Phasic Skin Conductance Response (SCR).
        - *"SCR_Onsets"*: the samples at which the onsets of the peaks occur, marked as "1" in a list of zeros.
        - *"SCR_Peaks"*: the samples at which the peaks occur, marked as "1" in a list of zeros.
        - *"SCR_Height"*: the SCR amplitude of the signal including the Tonic component. Note that cumulative effects of close- occurring SCRs might lead to an underestimation of the amplitude.
        - *"SCR_Amplitude"*: the SCR amplitude of the signal excluding the Tonic component.
        - *"SCR_RiseTime"*: the time taken for SCR onset to reach peak amplitude within the SCR.
        - *"SCR_Recovery"*: the samples at which SCR peaks recover (decline) to half amplitude, marked as "1" in a list of zeros.
    info : dict
        A dictionary containing the information of each SCR peak (see `eda_findpeaks()`).

    See Also
    --------
    eda_simulate, eda_clean, eda_phasic, eda_findpeaks, eda_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> eda_signal = nk.eda_simulate(duration=30, scr_number=5, drift=0.1, noise=0)
    >>> signals, info = nk.eda_process(eda_signal, sampling_rate=1000)
    >>> fig = nk.eda_plot(signals)
    >>> fig #doctest: +SKIP

    """
    # Preprocess
    eda_cleaned = eda_clean(eda_signal, sampling_rate=sampling_rate, method="neurokit")
    eda_decomposed = eda_phasic(eda_cleaned, sampling_rate=sampling_rate)

    # Find peaks
    peak_signal, info = eda_peaks(
        eda_decomposed["EDA_Phasic"].values, sampling_rate=sampling_rate, method="neurokit", amplitude_min=0.1
    )

    # Store
    signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_cleaned})

    signals = pd.concat([signals, eda_decomposed, peak_signal], axis=1)

    return signals, info

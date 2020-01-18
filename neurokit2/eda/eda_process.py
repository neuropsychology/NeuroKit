# -*- coding: utf-8 -*-
import pandas as pd

from .eda_clean import eda_clean
from .eda_phasic import eda_phasic
from .eda_findpeaks import eda_findpeaks



def eda_process(eda_signal, sampling_rate=1000):
    """Process Electrodermal Activity (EDA).

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
        - *"EDA_Phasic"*: the inhalation peaks marked as "1" in a list of zeros.
        - *"EDA_Tonic"*: the exhalation troughs marked as "1" in a list of
                            zeros.
        - *"SCR_Peaks"*: breathing rate interpolated between inhalation peaks.
    info : dict
        A dictionary containing the information of each SCR peak (see `eda_findpeaks()`).

    See Also
    --------
    eda_simulate, eda_clean, eda_phasic, eda_findpeaks, eda_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> eda_signal = nk.eda_simulate(duration=30, n_scr=5, drift=0.1, noise=0)
    >>> signals, info = nk.eda_process(eda_signal, sampling_rate=1000)
    >>> nk.eda_plot(signals)
    """
    # Preprocess
    eda_cleaned = eda_clean(eda_signal, sampling_rate=sampling_rate, method="neurokit")
    eda_decomposed = eda_phasic(eda_cleaned, sampling_rate=sampling_rate)

    # Find peaks
    peaks, info = eda_findpeaks(eda_decomposed["EDA_Phasic"].values, sampling_rate=sampling_rate, method="neurokit", amplitude_min=0.1)

    # Store
    signals = pd.DataFrame({"EDA_Raw": eda_signal,
                            "EDA_Clean": eda_cleaned})

    signals = pd.concat([signals, eda_decomposed, peaks], axis=1)

    return signals, info

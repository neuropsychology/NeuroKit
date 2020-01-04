# -*- coding: utf-8 -*-

from ..signal import signal_filter


def ecg_clean(ecg_signal, sampling_rate=1000, method="neurokit"):
    """Clean an ECG signal.

    Prepare a raw ECG signal for R-peak detection with the specified method.

    Parameters
    ----------
    ecg_signal : list, array or Series
        The raw ECG channel.
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.
    method : str
        The processing pipeline to apply. Can be one of 'neurokit' (default).

    Returns
    -------
    array
        Vector containing the cleaned ECG signal.

    See Also
    --------
    ecg_findpeaks, ecg_rate, ecg_process, ecg_plot

    Examples
    --------
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=10, sampling_rate=1000)
    >>> signals = pd.DataFrame({
            "RSP_Raw": ecg,
            "RSP_NeuroKit": nk.ecg_clean(ecg, sampling_rate=1000, method="neurokit")})
    >>> signals.plot()
    """
    method = method.lower()  # remove capitalised letters
    if method in ["neurokit", "nk"]:
        clean = _ecg_clean_nk(ecg_signal, sampling_rate=sampling_rate)
    else:
        raise ValueError("NeuroKit error: ecg_clean(): 'method' should be "
                         "one of 'neurokit'.")
    return clean





# =============================================================================
# Neurokit
# =============================================================================
def _ecg_clean_nk(signal, sampling_rate):

    # Remove slow drift and dc offset with highpass Butterworth.
    clean = signal_filter(signal=signal,
                          sampling_rate=sampling_rate,
                          lowcut=.5,
                          method="butterworth",
                          order=5)
    return clean

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
        The processing pipeline to apply. Defaultsd to "neurokit".

    Returns
    -------
    array
        Vector containing the cleaned ECG signal.

    See Also
    --------
    ecg_findpeaks, ecg_rate, ecg_process, ecg_plot

    Examples
    --------
    >>>
    """
    # Determine method and clean ECG.
    cleanfun = False
    method = method.lower()
    if method == "neurokit":
        cleanfun = _ecg_clean_nk
    if not cleanfun:
        print("NeuroKit error: Please choose a valid method.")
        return

    clean = cleanfun(signal=ecg_signal, sampling_rate=sampling_rate)

    return clean


def _ecg_clean_nk(signal, sampling_rate):

    # Remove slow drift and dc offset with highpass Butterworth.
    clean = signal_filter(signal=signal, sampling_rate=sampling_rate,
                          lowcut=.5, method="butterworth", order=5)
    return clean

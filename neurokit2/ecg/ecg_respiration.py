# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate

from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize
from ..signal import signal_detrend


def ecg_respiration(peaks, sampling_rate=1000, show=False):
    """Computes ECG-Derived Respiration (EDR).

    Generates respiratory signal from ECG signal using R-R intervals,
    as described in https://gist.github.com/
    raphaelvallat/55624e2eb93064ae57098dd96f259611.

    Parameters
    ----------
    peaks : dict
        The samples at which the R-peaks occur. Dict returned by
        `ecg_peaks()`.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    show : bool
        Plots the EDR signal. Defaults to False.

    Examples
    --------
    >>> ecg = nk.ecg_simulate(duration=30, sampling_rate=1000)
    >>> cleaned = nk.ecg_clean(ecg, sampling_rate=1000)
    >>> signal, peaks = nk.ecg_peaks(cleaned, correct_artifacts=True)
    >>> nk.ecg_respiration(peaks, show=True)

    Returns
    -------
    edr : array
        An array consisting of the (normalized) EDR signal.

    """
    # RR peaks detection
    if isinstance(peaks, tuple):
        raise ValueError("NeuroKit error: _ecg_respiration_formatinput(): Wrong"
                         "input. Please make sure only the dictionary of R peaks"
                         "is provided.")
    if isinstance(peaks, pd.DataFrame):
        peaks = _signal_formatpeaks_sanitize(peaks, desired_length=None)
    else:
        peaks = peaks["ECG_R_Peaks"]

    rri = np.diff(peaks) / sampling_rate * 1000

    # Interpolate RR intervals using cubic spline
    rri_time = np.cumsum(rri) / 1000
    time_rri = rri_time - rri_time[0]
    interpolate = np.arange(0, time_rri[-1], 1 / 4)
    tck = scipy.interpolate.splrep(time_rri, rri, s=0)
    rri_interpolate = scipy.interpolate.splev(interpolate, tck, der=0)

    heart_rate = 1000 * 60/rri_interpolate

    # Detrend, Normalize
    edr = signal_detrend(heart_rate)
    edr = (edr - edr.mean()) / edr.std()

    # Find RSP peaks
#    rsp_peaks_signal, rsp_peaks_info = nk.rsp_peaks(edr, sampling_rate = sampling_rate*4)
#    data = pd.concat([pd.DataFrame({"EDR": edr}), rsp_peaks_signal], axis=1)
#    nk.signal_plot(data)

    # Plot EDR
    if show:
        plt.plot(edr)
        plt.title("ECG-Derived Respiration")

    return edr

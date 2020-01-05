# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal

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
        The processing pipeline to apply. Can be one of 'neurokit' (default)
        or 'biosppy'.

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
            "ECG_Raw": ecg,
            "ECG_NeuroKit": nk.ecg_clean(ecg, sampling_rate=1000, method="neurokit"),
            "ECG_BioSPPy":nk.ecg_clean(ecg, sampling_rate=1000, method="biosppy")})
    >>> signals.plot()
    """
    method = method.lower()  # remove capitalised letters
    if method in ["neurokit", "nk"]:
        clean = _ecg_clean_nk(ecg_signal, sampling_rate=sampling_rate)
    elif method in ["biosppy"]:
        clean = _ecg_clean_biosppy(ecg_signal, sampling_rate=sampling_rate)
    else:
        raise ValueError("NeuroKit error: ecg_clean(): 'method' should be "
                         "one of 'neurokit' or 'biosppy'.")
    return clean





# =============================================================================
# Neurokit
# =============================================================================
def _ecg_clean_nk(signal, sampling_rate=1000):

    # Remove slow drift and dc offset with highpass Butterworth.
    clean = signal_filter(signal=signal,
                          sampling_rate=sampling_rate,
                          lowcut=0.5,
                          method="butterworth",
                          order=5)
    return clean



# =============================================================================
# Biosppy
# =============================================================================
def _ecg_clean_biosppy(signal, sampling_rate=1000):
    """
    adapted from https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L69
    """

    order = int(0.3 * sampling_rate)
    if order % 2 == 0:
        order += 1  # Enforce odd number

    # -> filter_signal()
    frequency = [3, 45]

    #   -> get_filter()
    #     -> _norm_freq()
    frequency = 2 * np.array(frequency) / sampling_rate  # Normalize frequency to Nyquist Frequency (Fs/2).

    #     -> get coeffs
    a = np.array([1])
    b = scipy.signal.firwin(numtaps=order, cutoff=frequency, pass_zero=False)

    # _filter_signal()
    filtered = scipy.signal.filtfilt(b, a, signal)

    return filtered

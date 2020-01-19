# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal

from ..signal import signal_filter
from ..signal import signal_detrend


def emg_clean(emg_signal, sampling_rate=1000, method="neurokit"):
    """Clean an EMG signal.

    Prepare a raw EMG signal for R-peak detection with the specified method.

    Parameters
    ----------
    emg_signal : list, array or Series
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
        Vector containing the cleaned EMG signal.

    See Also
    --------
    emg_amplitude, emg_process, emg_plot

    Examples
    --------
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> emg = nk.emg_simulate(duration=10, sampling_rate=1000)
    >>> signals = pd.DataFrame({
            "EMG_Raw": emg,
            "EMG_NeuroKit": nk.emg_clean(emg, sampling_rate=1000, method="neurokit"),
            "EMG_BioSPPy":nk.emg_clean(emg, sampling_rate=1000, method="biosppy")})
    >>> signals.plot()
    """
    method = method.lower()  # remove capitalised letters
    if method in ["neurokit", "nk"]:
        clean = _emg_clean_nk(emg_signal, sampling_rate=sampling_rate)
    elif method in ["biosppy"]:
        clean = _emg_clean_biosppy(emg_signal, sampling_rate=sampling_rate)
    else:
        raise ValueError("NeuroKit error: emg_clean(): 'method' should be "
                         "one of 'neurokit' or 'biosppy'.")
    return clean






# =============================================================================
# Neurokit
# =============================================================================
def _emg_clean_nk(emg_signal, sampling_rate=1000):

    clean = signal_filter(signal=emg_signal,
                          sampling_rate=sampling_rate,
                          lowcut=0.5,
                          method="butterworth",
                          order=4)
    return clean


# =============================================================================
# BioSPPy
# =============================================================================
def _emg_clean_biosppy(emg_signal, sampling_rate=1000):
    """Uses the same defaults as `BioSPPy
    <https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/emg.py>`_.
    """
    # Parameters
    order = 4
    frequency = 100
    frequency = 2 * np.array(frequency) / sampling_rate  # Normalize frequency to Nyquist Frequency (Fs/2).

    # Filtering
    b, a = scipy.signal.butter(N=order, Wn=frequency, btype='highpass', analog=False)
    filtered = scipy.signal.filtfilt(b, a, emg_signal)

    # Baseline detrending
    clean = signal_detrend(filtered, order=0)

    return clean

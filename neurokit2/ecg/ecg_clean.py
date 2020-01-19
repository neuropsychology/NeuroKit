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
        The processing pipeline to apply. Can be one of 'neurokit' (default),
        'biosppy' or 'pamtompkins1985'.

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
            "ECG_BioSPPy":nk.ecg_clean(ecg, sampling_rate=1000, method="biosppy"),
            "ECG_PanTompkins":nk.ecg_clean(ecg, sampling_rate=1000, method="pantompkins1985"),
            "ECG_Hamilton":nk.ecg_clean(ecg, sampling_rate=1000, method="hamilton2002")})
    >>> signals.plot()

    References
    --------------
    - Jiapu Pan and Willis J. Tompkins. A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering BME-32.3 (1985), pp. 230–236.
    - Hamilton, Open Source ECG Analysis Software Documentation, E.P.Limited, 2002.
    """
    method = method.lower()  # remove capitalised letters
    if method in ["nk", "nk2", "neurokit", "neurokit2"]:
        clean = _ecg_clean_nk(ecg_signal, sampling_rate)
    elif method in ["biosppy"]:
        clean = _ecg_clean_biosppy(ecg_signal, sampling_rate)
    elif method in ["pantompkins", "pantompkins1985"]:
        clean = _ecg_clean_pantompkins(ecg_signal, sampling_rate)
    elif method in ["hamilton", "hamilton2002"]:
        clean = _ecg_clean_hamilton(ecg_signal, sampling_rate)
    else:
        raise ValueError("NeuroKit error: ecg_clean(): 'method' should be "
                         "one of 'neurokit' or 'biosppy'.")
    return clean





# =============================================================================
# Neurokit
# =============================================================================
def _ecg_clean_nk(ecg_signal, sampling_rate=1000):

    # Remove slow drift and dc offset with highpass Butterworth.
    clean = signal_filter(signal=ecg_signal,
                          sampling_rate=sampling_rate,
                          lowcut=0.5,
                          method="butterworth",
                          order=5)
    return clean



# =============================================================================
# Biosppy
# =============================================================================
def _ecg_clean_biosppy(ecg_signal, sampling_rate=1000):
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
    filtered = scipy.signal.filtfilt(b, a, ecg_signal)

    return filtered


# =============================================================================
# Pan & Tompkins (1985)
# =============================================================================
def _ecg_clean_pantompkins(ecg_signal, sampling_rate=1000):
    """
    adapted from https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L69
    """

    f1 = 5/sampling_rate
    f2 = 15/sampling_rate
    order = 1

    b, a = scipy.signal.butter(order, [f1*2, f2*2], btype='bandpass')

    filtered = scipy.signal.lfilter(b, a, ecg_signal)

    return filtered



# =============================================================================
# Hamilton (2002)
# =============================================================================
def _ecg_clean_hamilton(ecg_signal, sampling_rate=1000):
    """
    adapted from https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L69
    """

    f1 = 8/sampling_rate
    f2 = 16/sampling_rate

    b, a = scipy.signal.butter(1, [f1*2, f2*2], btype='bandpass')

    filtered = scipy.signal.lfilter(b, a, ecg_signal)

    return filtered
# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal

from ..signal import signal_smooth


def eda_clean(rsp_signal, sampling_rate=1000, method="biosppy"):
    """Preprocess a respiration (RSP) signal.

    Clean a respiration signal using different sets of parameters, such as:

    - `Khodadad et al. (2018)
    <https://iopscience.iop.org/article/10.1088/1361-6579/aad7e6/meta>`_:
        linear detrending followed by a fifth order 2Hz low-pass IIR
        Butterworth filter).
    - `BioSPPy
    <https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/resp.py>`_:
        second order 0.1 - 0.35 Hz bandpass Butterworth filter followed by a
        constant detrending.

    Parameters
    ----------
    rsp_signal : list, array or Series
        The raw respiration channel (as measured, for instance, by a
        respiration belt).
    sampling_rate : int
        The sampling frequency of `rsp_signal` (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of "khodadad2018"
        (default) or "biosppy".

    Returns
    -------
    array
        Vector containing the cleaned respiratory signal.

    See Also
    --------
    rsp_findpeaks, rsp_rate, rsp_amplitude, rsp_process, rsp_plot

    Examples
    --------
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> eda = nk.eda_simulate(duration=30, sampling_rate=100, n_scr=10, noise=0.01, drift=0.02)
    >>> signals = pd.DataFrame({
            "EDA_Raw": eda,
            "EDA_BioSPPy": nk.eda_clean(eda, method='biosppy')})
    >>> signals.plot()
    """
    method = method.lower()  # remove capitalised letters
    if method == "biosppy":
        clean = _eda_clean_biosppy(rsp_signal, sampling_rate)
    else:
        raise ValueError("NeuroKit error: eda_clean(): 'method' should be "
                         "one of 'biosppy'.")

    return clean



# =============================================================================
# BioSPPy
# =============================================================================
def _eda_clean_biosppy(eda_signal, sampling_rate=1000):
    """Uses the same defaults as `BioSPPy
    <https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/eda.py>`_.
    """
    # Parameters
    order = 4
    frequency = 5
    frequency = 2 * np.array(frequency) / sampling_rate  # Normalize frequency to Nyquist Frequency (Fs/2).

    # Filtering
    b, a = scipy.signal.butter(N=order, Wn=frequency, btype='lowpass', analog=False, output='ba')
    filtered = scipy.signal.filtfilt(b, a, eda_signal)

    # Smoothing
    clean = signal_smooth(filtered, method='convolution', kernel='boxzen', size=int(0.75 * sampling_rate))

    return clean
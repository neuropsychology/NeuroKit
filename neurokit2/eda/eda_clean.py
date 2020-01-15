# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal

from ..signal import signal_smooth


def eda_clean(eda_signal, sampling_rate=1000, method="biosppy"):
    """Preprocess Electrodermal Activity (EDA) signal.


    Parameters
    ----------
    eda_signal : list, array or Series
        The raw EDA signal.
    sampling_rate : int
        The sampling frequency of `rsp_signal` (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of 'biosppy' (default).

    Returns
    -------
    array
        Vector containing the cleaned EDA signal.

    See Also
    --------
    eda_simulate, eda_decompose

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
        clean = _eda_clean_biosppy(eda_signal, sampling_rate)
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

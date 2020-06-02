# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal

from ..misc import as_vector
from ..signal import signal_detrend


def emg_clean(emg_signal, sampling_rate=1000):
    """
    Preprocess an electromyography (emg) signal.

    Clean an EMG signal using a set of parameters, such as:
    - `BioSPPy
    <https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/emg.py>>`_:
        fourth order 100 Hz highpass Butterworth filter followed by a
        constant detrending.

    Parameters
    ----------
    emg_signal : list, array or Series
        The raw EMG channel.
    sampling_rate : int
        The sampling frequency of `emg_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.
    method : str
        The processing pipeline to apply. Can be one of 'biosppy' (default).

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
    >>> signals = pd.DataFrame({"EMG_Raw": emg, "EMG_Cleaned":nk.emg_clean(emg, sampling_rate=1000)})
    >>> fig = signals.plot()
    >>> fig #doctest: +SKIP

    """
    emg_signal = as_vector(emg_signal)

    # Parameters
    order = 4
    frequency = 100
    frequency = 2 * np.array(frequency) / sampling_rate  # Normalize frequency to Nyquist Frequency (Fs/2).

    # Filtering
    b, a = scipy.signal.butter(N=order, Wn=frequency, btype="highpass", analog=False)
    filtered = scipy.signal.filtfilt(b, a, emg_signal)

    # Baseline detrending
    clean = signal_detrend(filtered, order=0)

    return clean

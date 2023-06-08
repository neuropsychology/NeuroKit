# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np
import pandas as pd
import scipy.signal

from ..misc import NeuroKitWarning, as_vector
from ..signal import signal_detrend


def emg_clean(emg_signal, sampling_rate=1000, method="biosppy"):
    """**Preprocess an electromyography (emg) signal**

    Clean an EMG signal using a set of parameters. Only one method is available at the moment.

    * **BioSPPy**: fourth order 100 Hz highpass Butterworth filter followed by a constant
      detrending.

    Parameters
    ----------
    emg_signal : Union[list, np.array, pd.Series]
        The raw EMG channel.
    sampling_rate : int
        The sampling frequency of ``emg_signal`` (in Hz, i.e., samples/second).
        Defaults to 1000.
    method : str
        The processing pipeline to apply. Can be one of ``"biosppy"`` or ``"none"``.
        Defaults to ``"biosppy"``. If ``"none"`` is passed, the raw signal will be returned without
        any cleaning.

    Returns
    -------
    array
        Vector containing the cleaned EMG signal.

    See Also
    --------
    emg_amplitude, emg_process, emg_plot

    Examples
    --------
    .. ipython:: python

      import pandas as pd
      import neurokit2 as nk

      emg = nk.emg_simulate(duration=10, sampling_rate=1000)
      signals = pd.DataFrame({"EMG_Raw": emg, "EMG_Cleaned":nk.emg_clean(emg, sampling_rate=1000)})
      @savefig p_emg_clean1.png scale=100%
      fig = signals.plot()
      @suppress
      plt.close()

    """
    emg_signal = as_vector(emg_signal)

    # Missing data
    n_missing = np.sum(np.isnan(emg_signal))
    if n_missing > 0:
        warn(
            "There are " + str(n_missing) + " missing data points in your signal."
            " Filling missing values by using the forward filling method.",
            category=NeuroKitWarning,
        )
        emg_signal = _emg_clean_missing(emg_signal)

    method = str(method).lower()
    if method in ["none"]:
        clean = emg_signal
    elif method in ["biosppy"]:
        clean = _emg_clean_biosppy(emg_signal, sampling_rate=sampling_rate)
    else:
        raise ValueError(
            "NeuroKit error: emg_clean(): 'method' should be one of 'biosppy' or 'none'."
        )
    return clean


# =============================================================================
# Handle missing data
# =============================================================================
def _emg_clean_missing(emg_signal):

    emg_signal = pd.DataFrame.pad(pd.Series(emg_signal))

    return emg_signal


# =============================================================================
# BioSPPy
# =============================================================================
def _emg_clean_biosppy(emg_signal, sampling_rate=1000):
    # Parameters
    order = 4
    frequency = 100
    frequency = (
        2 * np.array(frequency) / sampling_rate
    )  # Normalize frequency to Nyquist Frequency (Fs/2).

    # Filtering
    b, a = scipy.signal.butter(N=order, Wn=frequency, btype="highpass", analog=False)
    filtered = scipy.signal.filtfilt(b, a, emg_signal)

    # Baseline detrending
    clean = signal_detrend(filtered, order=0)

    return clean

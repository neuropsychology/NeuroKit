# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np
import pandas as pd
import scipy.signal

from ..misc import NeuroKitWarning, as_vector
from ..signal import signal_filter, signal_smooth


def eda_clean(eda_signal, sampling_rate=1000, method="neurokit"):
    """**Preprocess Electrodermal Activity (EDA) signal**

    This function cleans the EDA signal by removing noise and smoothing the signal with different methods.

    * **NeuroKit**: Default methods. Low-pass filter with a 3 Hz cutoff frequency and a 4th order
      Butterworth filter. Note thaht if the sampling rate is lower than 7 Hz (as it is the case
      with some signals recorded by wearables such as Empatica), the filtering is skipped (as there
      is no high enough frequency to remove).
    * **BioSPPy**: More aggresive filtering than NeuroKit's default method. Low-pass filter with a
      5 Hz cutoff frequency and a 4th order Butterworth filter.


    Parameters
    ----------
    eda_signal : Union[list, np.array, pd.Series]
        The raw EDA signal.
    sampling_rate : int
        The sampling frequency of `rsp_signal` (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of ``"neurokit"`` (default), ``"biosppy"``, or
        ``"none"``.

    Returns
    -------
    array
        Vector containing the cleaned EDA signal.

    See Also
    --------
    eda_simulate, eda_findpeaks, eda_process, eda_plot

    Examples
    --------
    .. ipython:: python

      import pandas as pd
      import neurokit2 as nk

      # Simulate raw signal
      eda = nk.eda_simulate(duration=15, sampling_rate=100, scr_number=10, noise=0.01, drift=0.02)

      # Clean
      eda_clean1 = nk.eda_clean(eda, sampling_rate=100, method='neurokit')
      eda_clean2 = nk.eda_clean(eda, sampling_rate=100, method='biosppy')

      @savefig p_eda_clean.png scale=100%
      nk.signal_plot([eda, eda_clean1, eda_clean2], labels=["Raw", "NeuroKit", "BioSPPy"])
      @suppress
      plt.close()

    """
    eda_signal = as_vector(eda_signal)

    # Missing data
    n_missing = np.sum(np.isnan(eda_signal))
    if n_missing > 0:
        warn(
            "There are " + str(n_missing) + " missing data points in your signal."
            " Filling missing values by using the forward filling method.",
            category=NeuroKitWarning,
        )
        eda_signal = _eda_clean_missing(eda_signal)

    method = method.lower()  # remove capitalised letters
    if method == "biosppy":
        clean = _eda_clean_biosppy(eda_signal, sampling_rate)
    elif method in ["default", "neurokit", "nk"]:
        clean = _eda_clean_neurokit(eda_signal, sampling_rate)
    elif method is None or method == "none":
        clean = eda_signal
    else:
        raise ValueError("NeuroKit error: eda_clean(): 'method' should be one of 'biosppy'.")

    return clean


# =============================================================================
# Handle missing data
# =============================================================================
def _eda_clean_missing(eda_signal):

    eda_signal = pd.DataFrame.pad(pd.Series(eda_signal))

    return eda_signal


# =============================================================================
# NK
# =============================================================================
def _eda_clean_neurokit(eda_signal, sampling_rate=1000):

    if sampling_rate <= 6:
        warn(
            "EDA signal is sampled at very low frequency. Skipping filtering.",
            category=NeuroKitWarning,
        )
        return eda_signal

    # Filtering
    filtered = signal_filter(
        eda_signal, sampling_rate=sampling_rate, highcut=3, method="butterworth", order=4
    )

    return filtered


# =============================================================================
# BioSPPy
# =============================================================================
def _eda_clean_biosppy(eda_signal, sampling_rate=1000):
    """Uses the same defaults as `BioSPPy.

    <https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/eda.py>`_.

    """
    # Parameters
    order = 4
    frequency = 5
    frequency = (
        2 * np.array(frequency) / sampling_rate
    )  # Normalize frequency to Nyquist Frequency (Fs/2).

    # Filtering
    b, a = scipy.signal.butter(N=order, Wn=frequency, btype="lowpass", analog=False, output="ba")
    filtered = scipy.signal.filtfilt(b, a, eda_signal)

    # Smoothing
    clean = signal_smooth(
        filtered, method="convolution", kernel="boxzen", size=int(0.75 * sampling_rate)
    )

    return clean

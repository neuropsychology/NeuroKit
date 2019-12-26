# -*- coding: utf-8 -*-
import pandas as pd

from ..signal import signal_detrend
from ..signal import signal_filter


def rsp_clean(rsp_signal, sampling_rate=1000, defaults="khodadad2018"):
    """Preprocess a respiration (RSP) signal.

    Respiration (RSP) signal cleaning using different sets of parameters,
    such as:

    - `Khodadad et al. (2018) <https://iopscience.iop.org/article/10.1088/1361-6579/aad7e6/meta>`_: linear detrending followed by a fifth order 2Hz low-pass IIR Butterworth filter).
    - `BioSPPy <https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/resp.py>`_: second order 0.1 - 0.35 Hz bandpass Butterworth filter.

    Parameters
    ----------
    rsp_signal : list, array or Series
        The raw respiration channel (as measured, for instance, by a
        respiration belt).
    sampling_rate : int
        The sampling frequency of rsp_signal (in Hz, i.e., samples/second).
    defaults : str
        The cleaning pipeline to apply. Can be one of 'khodadad2018' or 'biosppy'.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as the input signal containing the raw signal
        and the cleaned signal, accessible with the keys "RSP_Raw", and
        "RSP_Filtered" respectively.

    See Also
    --------
    rsp_findpeaks, rsp_rate, rsp_process, rsp_plot

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration=30, sampling_rate=50, noise=0.01)
    >>> signals = pd.DataFrame({
            "RSP_Raw": rsp,
            "RSP_Khodadad2018": nk.rsp_clean(rsp, sampling_rate=50, defaults="khodadad2018")})
    >>> signals.plot()
    """
    if defaults.lower() == "khodadad2018":
        filtered_rsp = _rsp_clean_khodadad2018(rsp_signal, sampling_rate)
    elif defaults.lower() == "biosppy":
        filtered_rsp = _rsp_clean_biosppy(rsp_signal, sampling_rate)
    else:
        raise ValueError("NeuroKit error: rsp_clean(): 'defaults' should be "
                         "one of 'khodadad2018' or 'biosppy'.")

    return filtered_rsp


# =============================================================================
# Khodadad et al. (2018)
# =============================================================================
def _rsp_clean_khodadad2018(rsp_signal, sampling_rate=1000):
    """The algorithm is based on (but not an exact
    implementation of) the "Zero-crossing algorithm with amplitude threshold"
    by `Khodadad et al. (2018) <https://iopscience.iop.org/article/10.1088/1361-6579/aad7e6/meta>`_.
    """
    # Detrend and lowpass-filter the signal to be able to reliably detect
    # zero crossings in raw signal.
    filtered_rsp = signal_detrend(rsp_signal, order=1)
    filtered_rsp = signal_filter(filtered_rsp, sampling_rate=sampling_rate,
                                 lowcut=None, highcut=2,
                                 method="butterworth", butterworth_order=5)
    return filtered_rsp


# =============================================================================
# BioSPPy
# =============================================================================
def _rsp_clean_biosppy(rsp_signal, sampling_rate=1000):
    """Uses the same defaults as `BioSPPy <https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/resp.py>`_.
    """
    filtered_rsp = signal_filter(rsp_signal, sampling_rate=sampling_rate,
                                 lowcut=0.1, highcut=0.35,
                                 method="butterworth", butterworth_order=2)
    return filtered_rsp

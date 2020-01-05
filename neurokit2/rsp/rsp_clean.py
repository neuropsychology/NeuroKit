# -*- coding: utf-8 -*-
from ..signal import signal_detrend
from ..signal import signal_filter


def rsp_clean(rsp_signal, sampling_rate=1000, method="khodadad2018"):
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
    >>> rsp = nk.rsp_simulate(duration=30, sampling_rate=50, noise=0.01)
    >>> signals = pd.DataFrame({
            "RSP_Raw": rsp,
            "RSP_Khodadad2018": nk.rsp_clean(rsp, sampling_rate=50, method="khodadad2018"),
            "RSP_BioSPPy": nk.rsp_clean(rsp, sampling_rate=50, method="biosppy")})
    >>> signals.plot()
    """
    method = method.lower()  # remove capitalised letters
    if method in ["khodadad", "khodadad2018"]:
        clean = _rsp_clean_khodadad2018(rsp_signal, sampling_rate)
    elif method == "biosppy":
        clean = _rsp_clean_biosppy(rsp_signal, sampling_rate)
    else:
        raise ValueError("NeuroKit error: rsp_clean(): 'method' should be "
                         "one of 'khodadad2018' or 'biosppy'.")

    return clean


# =============================================================================
# Khodadad et al. (2018)
# =============================================================================
def _rsp_clean_khodadad2018(rsp_signal, sampling_rate=1000):
    """The algorithm is based on (but not an exact
    implementation of) the "Zero-crossing algorithm with amplitude threshold"
    by `Khodadad et al. (2018)
    <https://iopscience.iop.org/article/10.1088/1361-6579/aad7e6/meta>`_.
    """
    # Detrend and lowpass-filter the signal to be able to reliably detect
    # zero crossings in raw signal.
    clean = signal_detrend(rsp_signal, order=1)
    clean = signal_filter(clean, sampling_rate=sampling_rate,
                          lowcut=None, highcut=2,
                          method="butterworth", order=5)

    return clean


# =============================================================================
# BioSPPy
# =============================================================================
def _rsp_clean_biosppy(rsp_signal, sampling_rate=1000):
    """Uses the same defaults as `BioSPPy
    <https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/resp.py>`_.
    """
    clean = signal_filter(rsp_signal, sampling_rate=sampling_rate,
                          lowcut=0.1, highcut=0.35,
                          method="butterworth", order=2)
    clean = signal_detrend(clean, order=0)

    return clean

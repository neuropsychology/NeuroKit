# -*- coding: utf-8 -*-
import numpy as np
import mne

import scipy.signal



def signal_filter(signal, sampling_rate=1000, lowcut=None, highcut=None, method="butterworth", order=2, window_length="default"):
    """Filter a signal using 'butterworth', 'fir' or 'savgol' filters.

    Apply a lowpass (if 'highcut' frequency is provided), highpass (if 'lowcut' frequency is provided) or bandpass (if both are provided) filter to the signal.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
        or "bandstop".
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    lowcut : float
        Lower cutoff frequency in Hz. The default is None.
    highcut : float
        Upper cutoff frequency in Hz. The default is None.
    method : str
        Can be one of 'butterworth', 'fir' or 'savgol'.
    order : int
        Only used if method is 'butterworth' or 'savgol'. Order of the filter (default is 2).
    window_length : int
        Only used if method is 'savgol'. The length of the filter window (i.e. the number of coefficients). Must be an odd integer. If 'default', will be set to the sampling rate divided by 10 (101 if the sampling rate is 1000 Hz).

    See Also
    --------
    signal_detrend, signal_psd

    Returns
    -------
    array
        Vector containing the filtered signal.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=50, num=10000)) # Low freq
    >>> signal += np.cos(np.linspace(start=0, stop=500, num=10000)) # High freq
    >>>
    >>> # Lowpass
    >>> pd.DataFrame({"Raw": signal,
                      "Butter_2": nk.signal_filter(signal, highcut=3, method='butterworth', order=2),
                      "Butter_3": nk.signal_filter(signal, highcut=3, method='butterworth', order=3),
                      "Butter_5": nk.signal_filter(signal, highcut=3, method='butterworth', order=5),
                      "FIR": nk.signal_filter(signal, highcut=3, method='fir')}).plot(subplots=True)

    >>># Highpass
    >>> pd.DataFrame({"Raw": signal,
                      "Butter_2": nk.signal_filter(signal, lowcut=2, method='butterworth', order=2),
                      "Butter_3": nk.signal_filter(signal, lowcut=2, method='butterworth', order=3),
                      "Butter_5": nk.signal_filter(signal, lowcut=2, method='butterworth', order=5),
                      "FIR": nk.signal_filter(signal, lowcut=2, method='fir')}).plot(subplots=True)

    >>> original = nk.rsp_simulate(duration=30, method="sinusoidal")
    >>> signal = original + np.cos(np.linspace(start=0, stop=15, num=len(original)))/2  # Low freq
    >>> signal += np.cos(np.linspace(start=0, stop=1000, num=len(signal)))/2  # High freq
    >>>
    >>> pd.DataFrame({"Raw": signal,
                      "Lowpass": nk.signal_filter(signal, highcut=2),
                      "Highpass": nk.signal_filter(signal, lowcut=0.1),
                      "Bandpass": nk.signal_filter(signal, lowcut=0.1, highcut=2),
                      "Savgol": nk.signal_filter(signal, method='savgol')}).plot(subplots=True)
    """
    # Sanity checks
    if lowcut is None and highcut is None:
        return signal

    method = method.lower()
    if method in ["sg", "savgol", "savitzky-golay"]:
        filtered = _signal_filter_savgol(signal, sampling_rate, order, window_length=window_length)
    elif method in ["butter", "butterworth"]:
        filtered = _signal_filter_butterworth(signal, sampling_rate, lowcut, highcut, order)
    elif method in ["fir"]:
        filtered = _signal_filter_fir(signal, sampling_rate, lowcut, highcut, window_length=window_length)
    else:
        raise ValueError("NeuroKit error: signal_filter(): 'method' should be "
                         "one of 'butterworth', 'savgol' or 'fir'.")
    return filtered


# =============================================================================
# Savitzky-Golay (savgol)
# =============================================================================

def _signal_filter_savgol(signal, sampling_rate=1000, order=2, window_length="default"):
    """Filter a signal using the Savitzky-Golay method.

    Default window size is chosen based on `Sadeghi, M., & Behnia, F. (2018). Optimum window length of Savitzky-Golay filters with arbitrary order. arXiv preprint arXiv:1808.10489. <https://arxiv.org/ftp/arxiv/papers/1808/1808.10489.pdf>`_.
    """
    window_length = _signal_filter_windowlength(window_length=window_length, sampling_rate=sampling_rate)

    filtered = scipy.signal.savgol_filter(signal, window_length=window_length, polyorder=order)
    return filtered

# =============================================================================
# FIR
# =============================================================================
def _signal_filter_fir(signal, sampling_rate=1000, lowcut=None, highcut=None, window_length="default"):
    """Filter a signal using a FIR filter.
    """
    if isinstance(window_length, str):
        window_length = "auto"

    filtered = mne.filter.filter_data(
            signal,
            sfreq=sampling_rate,
            l_freq=lowcut,
            h_freq=highcut,
            method='fir',
            fir_window='hamming',
            filter_length=window_length,
            l_trans_bandwidth='auto',
            h_trans_bandwidth='auto',
            phase='zero-double',
            fir_design='firwin',
            pad='reflect_limited',
            verbose=False)
    return filtered


# =============================================================================
# Butterworth
# =============================================================================

def _signal_filter_butterworth(signal, sampling_rate=1000, lowcut=None, highcut=None, order=5):
    """Filter a signal using IIR Butterworth SOS method.
    """
    freqs, filter_type = _signal_filter_sanitize(lowcut=lowcut, highcut=highcut)

    sos = scipy.signal.butter(order, freqs, btype=filter_type, output='sos', fs=sampling_rate)
    filtered = scipy.signal.sosfiltfilt(sos, signal)
    return filtered


# Old method, doesn't work great for bandpass filters

# def _signal_filter_butterworth_ba(signal, sampling_rate=1000, lowcut=None, highcut=None, butterworth_order=2):
#    """Filter a signal using IIR Butterworth B/A method.
#    """
#    # Get coefficients
#    nyquist_freq = 0.5 * sampling_rate
#
#    if lowcut is not None:
#        lowcut = lowcut / nyquist_freq
#    if highcut is not None:
#        highcut = highcut / nyquist_freq
#
#    if lowcut is not None and highcut is not None:
#        b, a = scipy.signal.butter(butterworth_order, [lowcut, highcut], btype="bandpass", output='ba')
#    elif lowcut is not None:
#        b, a = scipy.signal.butter(butterworth_order, [lowcut], btype="highpass", output='ba')
#    elif highcut is not None:
#        b, a = scipy.signal.butter(butterworth_order, [highcut], btype="lowpass", output='ba')
#
#    try:
#        filtered = scipy.signal.filtfilt(b, a, signal, method="gust")
#    except ValueError:
#        filtered = scipy.signal.filtfilt(b, a, signal, method="pad")
#
#    return filtered


# =============================================================================
# Internals
# =============================================================================
def _signal_filter_sanitize(lowcut=None, highcut=None):
    if lowcut is not None and highcut is not None:
        if lowcut > highcut:
            filter_type = "bandstop"
        else:
            filter_type = "bandpass"
        freqs = [lowcut, highcut]
    elif lowcut is not None:
        freqs = [lowcut]
        filter_type = "highpass"
    elif highcut is not None:
        freqs = [highcut]
        filter_type = "lowpass"

    return freqs, filter_type





def _signal_filter_windowlength(window_length="default", sampling_rate=1000):
    if isinstance(window_length, str):
        window_length = int(np.round(sampling_rate/3))
        if (window_length % 2) == 0:
            window_length + 1
    return window_length

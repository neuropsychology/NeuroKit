# -*- coding: utf-8 -*-
import scipy.signal



def signal_filter(signal, sampling_rate=1000, lowcut=None, highcut=None, method="butterworth", butterworth_order=5):
    """Filter a signal.

    Apply a lowpass (if 'highcut' frequency is provided), highpass (if 'lowcut' frequency is provided) or bandpass (if both are provided) filter to the signal.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
        or "bandstop".
    sampling_rate : int
        Sampling frequency (Hz).
    lowcut : float
        Lower cutoff frequency in Hz. The default is None.
    highcut : float
        Upper cutoff frequency in Hz. The default is None.
    method : str
        Can be one of 'butterworth'.
    butterworth_order : int
        Order of the filter when method is 'butterworth'. The default is 5.

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
    >>> signal = np.cos(np.linspace(start=0, stop=10, num=1000)) # Low freq
    >>> signal += np.cos(np.linspace(start=0, stop=100, num=1000)) # High freq
    >>> pd.DataFrame({"Raw": signal,
                      "Lowpass": nk.signal_filter(signal, highcut=10),
                      "Highpass": nk.signal_filter(signal, lowcut=2),
                      "Bandpass": nk.signal_filter(signal, lowcut=2, highcut=10)}).plot()

    >>> original = nk.rsp_simulate(duration=30, method="sinusoidal")
    >>> signal = original + np.cos(np.linspace(start=0, stop=15, num=len(original)))/2 # Low freq
    >>> signal += np.cos(np.linspace(start=0, stop=1000, num=len(signal)))/2 # High freq
    >>> pd.DataFrame({"Raw": signal,
                      "Lowpass": nk.signal_filter(signal, highcut=2),
                      "Highpass": nk.signal_filter(signal, lowcut=0.1),
                      "Bandpass": nk.signal_filter(signal, lowcut=0.1, highcut=2),
                      "Original": original}).plot()
    """
    # Sanity checks
    if lowcut is None and highcut is None:
        return signal

    if method.lower() in ["butterworth", "butter"]:
        filtered = _signal_filter_butterworth(signal, sampling_rate, lowcut, highcut, butterworth_order)

    return filtered



# =============================================================================
# Butterworth
# =============================================================================

def _signal_filter_butterworth(signal, sampling_rate=1000, lowcut=None, highcut=None, butterworth_order=2):
    """Filter a signal using IIR Butterworth SOS method.
    """
    if lowcut is not None and highcut is not None:
        sos = scipy.signal.butter(butterworth_order, [lowcut, highcut], btype="bandpass", output='sos', fs=sampling_rate)
    elif lowcut is not None:
        sos = scipy.signal.butter(butterworth_order, [lowcut], btype="highpass", output='sos', fs=sampling_rate)
    elif highcut is not None:
        sos = scipy.signal.butter(butterworth_order, [highcut], btype="lowpass", output='sos', fs=sampling_rate)

    filtered = scipy.signal.sosfiltfilt(sos, signal)
    return filtered


# Old method, doesn't work great for bandpass filters

#def _signal_filter_butterworth_ba(signal, sampling_rate=1000, lowcut=None, highcut=None, butterworth_order=2):
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

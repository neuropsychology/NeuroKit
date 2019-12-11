# -*- coding: utf-8 -*-
import scipy.signal



def signal_filter(signal, sampling_rate=1000, lowcut=None, highcut=None, method="butterworth", butterworth_order=5):
    """Filter a signal.

    Will apply a lowpass (if 'highcut' frequency is provided), highpass (if 'lowcut' frequency is provided) or bandpass (if both are provided) filter to the signal.

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
    >>> filtered = nk.signal_filter(signal, highcut=10)
    >>> pd.DataFrame({"Raw": signal, "Filtered": filtered}).plot()
    """
    # Sanity checks
    if lowcut is None and highcut is None:
        raise ValueError("NeuroKit error: signal_filter(): 'lowcut' or " \
                         "'highcut' frequencies must be provided for filtering.")

    if method.lower() in ["butterworth", "butter"]:
        filtered = _signal_filter_butterworth(signal, sampling_rate, lowcut, highcut, butterworth_order)

    return(filtered)



# =============================================================================
# Butterworth
# =============================================================================

def _signal_filter_butterworth(signal, sampling_rate=1000, lowcut=None, highcut=None, butterworth_order=5):
    """Filter a signal using IIR Butterworth coefficients.
    """
    # Use filtfilt to obtain a zero-phase filter in order for the filtered
    # signal to not be phase shifted with respect to the original signal.
    b, a = _signal_filter_butterworth_coefficients(sampling_rate, lowcut, highcut, butterworth_order)
    filtered = scipy.signal.filtfilt(b, a, signal, method="gust")
    return(filtered)


def _signal_filter_butterworth_coefficients(sampling_rate=1000, lowcut=None, highcut=None, butterworth_order=5):
    """Calculate IIR Butterworth coefficient.

    Returns
    -------
    b, a : ndarray
        Numerator and Denominator polynomials of the filter.
    """
    nyquist_freq = 0.5 * sampling_rate

    if lowcut is not None:
        lowcut = lowcut / nyquist_freq
    if highcut is not None:
        highcut = highcut / nyquist_freq

    if lowcut is not None and highcut is not None:
        b, a = scipy.signal.butter(butterworth_order, [lowcut, highcut], btype="bandpass")
    elif lowcut is not None:
        b, a = scipy.signal.butter(butterworth_order, [lowcut], btype="highpass")
    elif highcut is not None:
        b, a = scipy.signal.butter(butterworth_order, [highcut], btype="lowpass")

    return(b, a)

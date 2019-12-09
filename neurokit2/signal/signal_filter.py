# -*- coding: utf-8 -*-

from scipy.signal import butter, filtfilt


def butter_filter(x, filttype, sfreq, lowcut=None, highcut=None, order=5):
    """Filter a signal using IIR Butterworth coefficients.

    Parameters
    ----------
    x : 1d array
        Input signal.
    filttype : string
        Type of the filter. Can be either "lowpass", "highpass", "bandpass",
        or "bandstop".
    sfreq : int
        Sampling frequency (Hz).
    lowcut : float, optional
        Lower cutoff frequency in Hz. The default is None.
    highcut : float, optional
        Upper cutoff frequency in Hz. The default is None.
    order : int, optional
        Order of the filter. The default is 5.

    Returns
    -------
    y : 1d array
        Filtered signal.

    """
    # Use filtfilt to obtain a zero-phase filter in order for the filtered
    # signal to not be phase shifted with respect to the original signal.
    b, a = _butter_coefficients(filttype, sfreq, lowcut, highcut, order)
    y = filtfilt(b, a, x, method="pad")
    return y


def _butter_coefficients(filttype, sfreq, lowcut, highcut, order):
    """Calculate IIR Butterworth coefficient.

    Parameters
    ----------
    filttype : string
        Type of the filter. Can be either "lowpass", "highpass", "bandpass",
        or "bandstop".
    sfreq : int
        Sampling frequency (Hz).
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    order : int
        Order of the filter.

    Returns
    -------
    b : ndarray
        Numerator polynomials of the filter.
    a : ndarray
        Denominator polynomials of the filter.

    """
    nyq = 0.5 * sfreq

    if lowcut:
        lowcut_norm = lowcut / nyq
    if highcut:
        highcut_norm = highcut / nyq

    if filttype == "bandpass" or filttype == "bandstop":
        Wn = [lowcut_norm, highcut_norm]
    elif filttype == "lowpass":
        Wn = [highcut_norm]
    elif filttype == "highpass":
        Wn = [lowcut_norm]

    b, a = butter(order, Wn, btype=filttype)
    return b, a

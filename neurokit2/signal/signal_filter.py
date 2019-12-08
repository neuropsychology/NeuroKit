# -*- coding: utf-8 -*-

from scipy.signal import butter, filtfilt


def butter_filter(x, filttype, sfreq, **kwargs):
    """
    Use filtfilt to obtain a zero-phase filter, i.e. the filtered signal is not
    phase shifted with respect to the original signal since the filtering is
    performed in both directions:

    returns:
        filtered signal
    """

    b, a = _butter_coefficients(filttype, sfreq, **kwargs)
    y = filtfilt(b, a, x, method="pad")
    return y


def _butter_coefficients(filttype, sfreq, lowcut=None, highcut=None, order=5):
    """
    returns:
        filter coefficients for the requested filter type
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

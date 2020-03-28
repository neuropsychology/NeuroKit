# -*- coding: utf-8 -*-

from ..signal import signal_filter
from ..misc import sanitize_input


def ppg_clean(ppg_signal, sampling_rate=1000, method="elgendi"):
    """Clean a photoplethysmogram (PPG) signal.

    Prepare a raw PPG signal for systolic peak detection.

    Parameters
    ----------
    ppg_signal : list, array or Series
        The raw PPG channel.
    sampling_rate : int
        The sampling frequency of the PPG (in Hz, i.e., samples/second). The
        default is 1000.
    method : str
        The processing pipeline to apply. Can be one of "elgendi". The default
        is "elgendi".

    Returns
    -------
    clean : array
        A vector containing the cleaned PPG.

    See Also
    --------
    ppg_findpeaks

    Examples
    --------
    >>> import neurokit2 as nk

    """
    ppg_signal = sanitize_input(ppg_signal,
                                message="NeuroKit error: ppg_clean(): Please"
                                " provide the signal in vector form, i.e., a"
                                " one-dimensional array (e.g., a list).")

    method = method.lower()
    if method in ["elgendi"]:
        clean = _ppg_clean_elgendi(ppg_signal, sampling_rate)
    else:
        raise ValueError("Neurokit error: Please use one of the following"
                         " methods: 'elgendi'.")

    return clean


def _ppg_clean_elgendi(ppg_signal, sampling_rate):

    filtered = signal_filter(ppg_signal, sampling_rate=sampling_rate,
                             lowcut=.5, highcut=8, order=3, method="butter_ba")
    return filtered

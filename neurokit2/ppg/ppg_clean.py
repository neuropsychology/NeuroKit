# -*- coding: utf-8 -*-

from ..misc import as_vector
from ..signal import signal_filter


def ppg_clean(ppg_signal, sampling_rate=1000, method="elgendi"):
    """
    Clean a photoplethysmogram (PPG) signal.

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
    ppg_simulate, ppg_findpeaks

    Examples
    --------
    >>> import neurokit2 as nk
    >>> import matplotlib.pyplot as plt
    >>>
    >>> ppg = nk.ppg_simulate(heart_rate=75, duration=30)
    >>> ppg_clean = nk.ppg_clean(ppg)
    >>>
    >>> plt.plot(ppg, label="raw PPG") #doctest: +SKIP
    >>> plt.plot(ppg_clean, label="clean PPG") #doctest: +SKIP
    >>> plt.legend() #doctest: +SKIP

    """
    ppg_signal = as_vector(ppg_signal)

    method = method.lower()
    if method in ["elgendi"]:
        clean = _ppg_clean_elgendi(ppg_signal, sampling_rate)
    else:
        raise ValueError("Neurokit error: Please use one of the following methods: 'elgendi'.")

    return clean


def _ppg_clean_elgendi(ppg_signal, sampling_rate):

    filtered = signal_filter(
        ppg_signal, sampling_rate=sampling_rate, lowcut=0.5, highcut=8, order=3, method="butter_ba"
    )
    return filtered

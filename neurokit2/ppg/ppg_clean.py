# -*- coding: utf-8 -*-

from ..misc import as_vector
from ..signal import signal_filter


def ppg_clean(ppg_signal, sampling_rate=1000, method="elgendi"):
    """Clean a photoplethysmogram (PPG) signal.

    Prepare a raw PPG signal for systolic peak detection.

    Parameters
    ----------
    ppg_signal : Union[list, np.array, pd.Series]
        The raw PPG channel.
    sampling_rate : int
        The sampling frequency of the PPG (in Hz, i.e., samples/second). The default is 1000.
    method : str
        The processing pipeline to apply. Can be one of "elgendi" or "nabian2018". The default is "elgendi".
    highcut : int
        The cut-off frequency for removal of high frequency artifacts in BP signals (in Hz, i.e., samples/second). The default is 40.

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

    References
    ----------
    - Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., &amp; Ostadabbas, S. (2018). An Open-Source 
    Feature Extraction Tool for the Analysis of Peripheral Physiological Data. IEEE Journal of Translational Engineering in 
    Health and Medicine, 6, 1-11. doi:10.1109/jtehm.2018.2878000
    """
    ppg_signal = as_vector(ppg_signal)

    method = method.lower()
    if method in ["elgendi"]:
        clean = _ppg_clean_elgendi(ppg_signal, sampling_rate)
    elif method in ["nabian2018"]:
        clean = _ppg_clean_nabian2018(ppg_signal, sampling_rate, highcut=highcut)
    else:
        raise ValueError("Neurokit error: Please use one of the following methods: 'elgendi' or 'nabian2018'.")

    return clean


def _ppg_clean_elgendi(ppg_signal, sampling_rate):

    filtered = signal_filter(
        ppg_signal, sampling_rate=sampling_rate, lowcut=0.5, highcut=8, order=3, method="butter_ba"
    )
    return filtered

def _ppg_clean_nabian2018 (ppg_signal, sampling_rate, highcut="default"):
    
    """Low-pass filter for continuous BP signal preprocessing"""

    filtered = signal_filter(ppg_signal, sampling_rate=sampling_rate, lowcut=None, highcut="default", order=2, method="butterworth")
    
    if highcut == "default":
        highcut == 40
        
    if highcut is not "default":
        highcut>=(10*heart_rate) and highcut<(0.5*sampling_rate)
    else:
        raise ValueError("Neurokit error: Highcut value should be at least 10 times heart rate and less than 0.5 times sampling rate.")
    
    return filtered
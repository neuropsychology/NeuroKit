# -*- coding: utf-8 -*-
from ..misc import as_vector
from ..signal import signal_filter


def ppg_clean(ppg_signal, sampling_rate=1000, heart_rate=None, method="elgendi"):
    """Clean a photoplethysmogram (PPG) signal.

    Prepare a raw PPG signal for systolic peak detection.

    Parameters
    ----------
    ppg_signal : Union[list, np.array, pd.Series]
        The raw PPG channel.
    heart_rate : Union[int, float]
        The heart rate of the PPG signal. Applicable only if method is "nabian2018" to check that
        filter frequency is appropriate.
    sampling_rate : int
        The sampling frequency of the PPG (in Hz, i.e., samples/second). The default is 1000.
    method : str
        The processing pipeline to apply. Can be one of "elgendi" or "nabian2018". The default is "elgendi".

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
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # Simulate and clean signal
    >>> ppg = nk.ppg_simulate(heart_rate=75, duration=30)
    >>> ppg_elgendi = nk.ppg_clean(ppg, method='elgendi')
    >>> ppg_nabian = nk.ppg_clean(ppg, method='nabian2018', heart_rate=75)
    >>>
    >>> # Plot and compare methods
    >>> signals = pd.DataFrame({"PPG_Raw" : ppg,
    ...                         "PPG_Elgendi" : ppg_elgendi,
    ...                         "PPG_Nabian" : ppg_nabian})
    >>> signals.plot() #doctest: +ELLIPSIS
     <AxesSubplot:>

    References
    ----------
    - Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., &amp; Ostadabbas, S. (2018).
    An Open-Source Feature Extraction Tool for the Analysis of Peripheral Physiological Data. IEEE Journal of
    Translational Engineering in Health and Medicine, 6, 1-11. doi:10.1109/jtehm.2018.2878000

    """
    ppg_signal = as_vector(ppg_signal)

    method = method.lower()
    if method in ["elgendi"]:
        clean = _ppg_clean_elgendi(ppg_signal, sampling_rate)
    elif method in ["nabian2018"]:
        clean = _ppg_clean_nabian2018(ppg_signal, sampling_rate, heart_rate=heart_rate)
    else:
        raise ValueError("Neurokit error: Please use one of the following methods: 'elgendi' or 'nabian2018'.")

    return clean

# =============================================================================
# Methods
# =============================================================================

def _ppg_clean_elgendi(ppg_signal, sampling_rate):

    filtered = signal_filter(
        ppg_signal, sampling_rate=sampling_rate, lowcut=0.5, highcut=8, order=3, method="butter_ba"
    )
    return filtered


def _ppg_clean_nabian2018(ppg_signal, sampling_rate, heart_rate=None):
    """Low-pass filter for continuous BP signal preprocessing, adaopted from Nabian et al. (2018).
    """

    # Determine low-pass filter value
    highcut = 40

    # Convert heart rate to seconds, check if low-pass filter within appropriate range
    if heart_rate is not None:
        heart_rate = heart_rate / 60

        if not highcut >= 10*heart_rate and not highcut < 0.5*sampling_rate:
            raise ValueError("Neurokit error: Highcut value should be at least 10 times heart rate and"
                             " less than 0.5 times sampling rate.")

    filtered = signal_filter(ppg_signal, sampling_rate=sampling_rate, lowcut=None, highcut=highcut, order=2, method="butterworth")

    return filtered

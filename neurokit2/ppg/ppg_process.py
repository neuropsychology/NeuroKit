# -*- coding: utf-8 -*-
import pandas as pd

from ..misc import as_vector
from ..signal import signal_rate
from ..signal.signal_formatpeaks import _signal_from_indices
from .ppg_clean import ppg_clean
from .ppg_findpeaks import ppg_findpeaks


def ppg_process(ppg_signal, sampling_rate=1000, **kwargs):
    """Process a photoplethysmogram (PPG)  signal.

    Convenience function that automatically processes a photoplethysmogram signal.

    Parameters
    ----------
    ppg_signal : Union[list, np.array, pd.Series]
        The raw PPG channel.
    sampling_rate : int
        The sampling frequency of `emg_signal` (in Hz, i.e., samples/second).

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as `emg_signal` containing the following columns:
        - *"PPG_Raw"*: the raw signal.
        - *"PPG_Clean"*: the cleaned signal.
        - *"PPG_Rate"*: the heart rate as measured based on PPG peaks.
        - *"PPG_Peaks"*: the PPG peaks marked as "1" in a list of zeros.

    info : dict
        A dictionary containing the information of peaks.

    See Also
    --------
    ppg_clean, ppg_findpeaks

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> ppg = nk.ppg_simulate(duration=10, sampling_rate=1000, heart_rate=70)
    >>> signals, info = nk.ppg_process(ppg, sampling_rate=1000)
    >>> fig = nk.ppg_plot(signals)
    >>> fig #doctest: +SKIP

    """
    # Sanitize input
    ppg_signal = as_vector(ppg_signal)

    # Clean signal
    ppg_cleaned = ppg_clean(ppg_signal, sampling_rate=sampling_rate)

    # Find peaks
    info = ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate, **kwargs)

    # Mark peaks
    peaks_signal = _signal_from_indices(info["PPG_Peaks"], desired_length=len(ppg_cleaned))

    # Rate computation
    rate = signal_rate(info["PPG_Peaks"], sampling_rate=sampling_rate, desired_length=len(ppg_cleaned))

    # Prepare output
    signals = pd.DataFrame(
        {"PPG_Raw": ppg_signal, "PPG_Clean": ppg_cleaned, "PPG_Rate": rate, "PPG_Peaks": peaks_signal}
    )

    return signals, info

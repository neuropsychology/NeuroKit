# -*- coding: utf-8 -*-
import pandas as pd

from .emg_activation import emg_activation
from .emg_amplitude import emg_amplitude
from .emg_clean import emg_clean


def emg_process(emg_signal, sampling_rate=1000):
    """
    Process a electromyography (EMG) signal.

    Convenience function that automatically processes
    an electromyography signal.

    Parameters
    ----------
    emg_signal : list, array or Series
        The raw electromyography channel.
    sampling_rate : int
        The sampling frequency of `emg_signal` (in Hz, i.e., samples/second).

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as `emg_signal` containing the following
        columns:

        - *"EMG_Raw"*: the raw signal.
        - *"EMG_Clean"*: the cleaned signal.
        - *"EMG_Amplitude"*: the signal amplitude,
        or the activation level of the signal.
        - *"EMG_Activity*": the activity of the signal for which amplitude
        exceeds the threshold specified, marked as "1" in a list of zeros.
        - *"EMG_Onsets"*: the onsets of the amplitude,
        marked as "1" in a list of zeros.
        - *"EMG_Offsets"*: the offsets of the amplitude,
        marked as "1" in a list of zeros.
    info : dict
        A dictionary containing the information of each
        amplitude onset, offset, and peak activity (see `emg_activation()`).

    See Also
    --------
    emg_clean, emg_amplitude, emg_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> emg = nk.emg_simulate(duration=10, sampling_rate=1000, burst_number=3)
    >>> signals, info = nk.emg_process(emg, sampling_rate=1000)
    >>> fig = nk.emg_plot(signals)
    >>> fig #doctest: +SKIP

    """
    # Clean signal
    emg_cleaned = emg_clean(emg_signal, sampling_rate=sampling_rate)

    # Get amplitude
    amplitude = emg_amplitude(emg_cleaned)

    # Get onsets, offsets, and periods of activity
    activity_signal, info = emg_activation(amplitude, sampling_rate=sampling_rate, threshold="default")

    # Prepare output
    signals = pd.DataFrame({"EMG_Raw": emg_signal, "EMG_Clean": emg_cleaned, "EMG_Amplitude": amplitude})

    signals = pd.concat([signals, activity_signal], axis=1)

    return signals, info

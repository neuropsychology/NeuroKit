# -*- coding: utf-8 -*-
import pandas as pd

from .emg_clean import emg_clean
from .emg_amplitude import emg_amplitude


def emg_process(emg_signal, sampling_rate=1000):
    """Process a electromyography (EMG) signal.

    Convenience function that automatically processes a electromyography signal.

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
        - *"EMG_Amplitude"*: the signal amplitude, or the activation level of the signal.

    See Also
    --------
    emg_clean, emg_amplitude, emg_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> emg = emg_simulate(duration=10, sampling_rate=1000, n_bursts=3)
    >>> signals = emg_process(emg, sampling_rate=1000)
    >>> emg_plot(signals)
    """
    # Clean signal
    emg_cleaned = emg_clean(emg_signal, sampling_rate=sampling_rate)

    # Get amplitude
    amplitude = emg_amplitude(emg_cleaned)

    # Prepare output
    signals = pd.DataFrame({"EMG_Raw": emg_signal,
                            "EMG_Clean": emg_cleaned,
                            "EMG_Amplitude": amplitude})

    return signals
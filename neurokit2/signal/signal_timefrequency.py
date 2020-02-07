# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal


def signal_psd(signal, sampling_rate=1000, show=True):
    """Compute the Power Spectral Density (PSD).

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    show : bool
        If True, will return a plot. If False, will return the density values that can be plotted externally.
    method : str
        Either 'mne' (default; using multitapers) or 'scipy' (for the Welch's method).

    See Also
    --------
    signal_filter, mne.time_frequency.psd_array_multitaper, scipy.signal.welch

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Power Spectrum values and a plot if
        `show` is True.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(frequency=1) + 0.5*nk.signal_simulate(frequency=4)
    >>> nk.signal_plot(signal)
    >>> nk.signal_psd(signal, method="mne")
    >>> nk.signal_psd(signal, method="scipy")
    """
    # Do something
    return signal

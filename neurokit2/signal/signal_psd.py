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

    See Also
    --------
    signal_filter

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Power Spectrum values and a plot if
        `show` is True.

    Examples
    --------
    >>> import scipy.misc
    >>> import neurokit2 as nk
    >>>
    >>> signal = scipy.misc.electrocardiogram()
    >>> nk.signal_plot(signal)
    >>> nk.signal_psd(signal, sampling_rate=360, show=True)
    """
    frequency, power = scipy.signal.welch(signal,
                                          fs=sampling_rate,
                                          scaling='spectrum',
                                          detrend=False,
                                          average='median')

    data = pd.DataFrame({"Frequency": frequency,
                         "Power": power})

    if show is True:
        ax = data.plot(x="Frequency", y="Power", logy=True, title='Power Spectral Density (PSD)')
        ax.set(xlabel="Frequency (Hz)", ylabel="Spectrum")
        return ax
    else:
        return data

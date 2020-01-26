# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal


def signal_psd(signal, sampling_rate=1000, method="mne", show=True):
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

    # MNE
    if method == "mne":
        try:
            import mne
            power, frequency = mne.time_frequency.psd_array_multitaper(signal,
                                                                       sfreq=sampling_rate,
                                                                       fmin=0,
                                                                       fmax=np.inf,
                                                                       adaptive=False,
                                                                       normalization='full',
                                                                       verbose=False)
        except ImportError:
            print("NeuroKit warning: signal_psd(): the 'mne'",
                  "module is required for the 'mne' method to run.",
                  "Please install it first (`pip install mne`). In",
                  "the meantime, 'method' has been set to 'scipy'.")
            method = "scipy"

    # Scipy
    if method == "scipy":
        frequency, power = scipy.signal.welch(signal,
                                              fs=sampling_rate,
                                              scaling='spectrum',
                                              detrend=False,
                                              average='median')

    # Store results
    data = pd.DataFrame({"Frequency": frequency,
                         "Power": power})

    if show is True:
        ax = data.plot(x="Frequency", y="Power", logy=True, title='Power Spectral Density (PSD)')
        ax.set(xlabel="Frequency (Hz)", ylabel="Spectrum")
        return ax
    else:
        return data



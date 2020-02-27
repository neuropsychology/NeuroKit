# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal


def signal_psd(signal, sampling_rate=1000, method="multitapers", show=True, min_frequency=0, max_frequency=np.inf):
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
        Either 'multitapers' (default; requires the 'mne' package), 'burg' (requires the 'spectrum' package) or 'welch' (requires the 'scipy' package).
    min_frequency, max_frequency : float
        The minimum and maximum frequencies.

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
    >>> signal = nk.signal_simulate(frequency=5) + 0.5*nk.signal_simulate(frequency=20)
    >>> nk.signal_plot(signal)
    >>> nk.signal_psd(signal, method="multitapers")
    >>> nk.signal_psd(signal, method="welch")
    """
    # Constant Detrend
    signal = signal - np.mean(signal)

    # MNE
    if method.lower() in ["multitapers", "mne"]:
        try:
            import mne
            power, frequency = mne.time_frequency.psd_array_multitaper(signal,
                                                                       sfreq=sampling_rate,
                                                                       fmin=min_frequency,
                                                                       fmax=max_frequency,
                                                                       adaptive=False,
                                                                       normalization='full',
                                                                       verbose=False)
        except ImportError:
            print("NeuroKit warning: signal_psd(): the 'mne'",
                  "module is required for the 'mne' method to run.",
                  "Please install it first (`pip install mne`). For now,",
                  "'method' has been set to 'scipy'.")
            method = "scipy"

    # BURG
    if method.lower() in ["burg", "pburg", "spectrum"]:
        try:
            import spectrum
            burg = spectrum.pburg(data=signal, order=16, NFFT=None, sampling=sampling_rate)
            burg.scale_by_freq = False
            burg()
            power = np.array(burg.frequencies())
            frequency = burg.psd
        except ImportError:
            print("NeuroKit warning: signal_psd(): the 'spectrum'",
                  "module is required for the 'burg' method to run.",
                  "Please install it first (`pip install spectrum`). For now,",
                  "'method' has been set to 'scipy'.")
            method = "scipy"

    # Scipy
    if method.lower() in ["scipy", "welch"]:
        frequency, power = scipy.signal.welch(signal,
                                              fs=sampling_rate,
                                              scaling='density',
                                              detrend=False,
                                              average='median')

    # Store results
    data = pd.DataFrame({"Frequency": frequency,
                         "Power": power})

    # Filter
    data = data.loc[np.logical_and(data["Frequency"] >= min_frequency, data["Frequency"] <= max_frequency)]

    if show is True:
        ax = data.plot(x="Frequency", y="Power", logy=True, title='Power Spectral Density (PSD)')
        ax.set(xlabel="Frequency (Hz)", ylabel="Spectrum")
        return ax
    else:
        return data

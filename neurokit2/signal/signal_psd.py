# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal


def signal_psd(signal, sampling_rate=1000, method="multitapers", show=True, min_frequency=0, max_frequency=np.inf, precision=2**12):
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
        Either 'multitapers' (default; requires the 'mne' package), or 'welch' (requires the 'scipy' package).
    min_frequency, max_frequency : float
        The minimum and maximum frequencies.
    precision : int
        The precision, used for the Welch method only (should be the power of 2).

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
    >>>
    >>> nk.signal_psd(signal, method="multitapers")
    >>> nk.signal_psd(signal, method="welch")
    >>>
    >>> data = nk.signal_psd(signal, method="multitapers", max_frequency=30, show=False)
    >>> data.plot(x="Frequency", y="Power")
    >>> data = nk.signal_psd(signal, method="welch", max_frequency=30, show=False)
    >>> data.plot(x="Frequency", y="Power")
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
                  "'method' has been set to 'welch'.")
            method = "welch"

    # Scipy
    if method.lower() not in ["multitapers", "mne", "burg", "pburg", "spectrum"]:

        if max_frequency == np.inf:
            max_frequency = int(sampling_rate / 2)

        if (max_frequency - min_frequency) != (sampling_rate / 2):
            ratio = (sampling_rate / 2) / (max_frequency - min_frequency)
            precision = int(precision * ratio)

        if precision > len(signal) / 2:
            precision = int(len(signal) / 2)


        frequency, power = scipy.signal.welch(signal,
                                              fs=sampling_rate,
                                              scaling='density',
                                              detrend=False,
                                              average='median',
                                              window=scipy.signal.windows.hann(precision*2, False),
                                              nfft=precision*2)

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

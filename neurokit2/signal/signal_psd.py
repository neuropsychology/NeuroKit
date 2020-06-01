# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal


def signal_psd(
    signal, sampling_rate=1000, method="welch", show=True, min_frequency=0, max_frequency=np.inf, window=None
):
    """
    Compute the Power Spectral Density (PSD).

    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    show : bool
        If True, will return a plot. If False, will return the density values that can be plotted externally.
    method : str
        Either 'multitapers' (default; requires the 'mne' package), or 'welch' (requires the 'scipy' package).
    min_frequency, max_frequency : float
        The minimum and maximum frequencies.
    window : int
        Length of each window in seconds (for Welch method).
    resolution : int
        Resolution is used to adjust the window length in Welch method. It is also balance between frequency resolution and temporal resolution since the short the window length, the higher the temporal resolution and the lower the frequency resolution, vice versa.

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
    >>> fig1 = nk.signal_psd(signal, method="multitapers")
    >>> fig1 #doctest: +SKIP
    >>> fig2 = nk.signal_psd(signal, method="welch", min_frequency=1)
    >>> fig2 #doctest: +SKIP
    >>>
    >>> data = nk.signal_psd(signal, method="multitapers", max_frequency=30, show=False)
    >>> fig3 = data.plot(x="Frequency", y="Power")
    >>> fig3 #doctest: +SKIP
    >>> data = nk.signal_psd(signal, method="welch", max_frequency=30, show=False, min_frequency=1)
    >>> fig4 = data.plot(x="Frequency", y="Power")
    >>> fig4 #doctest: +SKIP

    """
    # Constant Detrend
    signal = signal - np.mean(signal)

    # MNE
    if method.lower() in ["multitapers", "mne"]:
        try:
            import mne

            power, frequency = mne.time_frequency.psd_array_multitaper(
                signal,
                sfreq=sampling_rate,
                fmin=min_frequency,
                fmax=max_frequency,
                adaptive=True,
                normalization="full",
                verbose=False,
            )
        except ImportError:
            raise ImportError(
                "NeuroKit warning: signal_psd(): the 'mne'",
                "module is required for the 'mne' method to run.",
                "Please install it first (`pip install mne`).",
            )

    # BURG
    elif method.lower() in ["burg", "pburg", "spectrum"]:
        raise ValueError("NeuroKit warning: signal_psd(): the 'BURG' method has not been yet implemented.")

    # Welch (Scipy)
    else:
        # Define window length
        if min_frequency == 0:
            min_frequency = 0.001  # sanitize lowest frequency
        if window is not None:
            nperseg = int(window * sampling_rate)
        else:
            # to capture at least 2 cycles of min_frequency
            nperseg = int((2 / min_frequency) * sampling_rate)

        # in case duration of recording is not sufficient
        if nperseg > len(signal) / 2:
            print(
                "Neurokit warning: signal_psd(): The duration of recording is too short to support a sufficiently long window for high frequency resolution. Consider using a longer recording or increasing the `min_frequency`"
            )
            nperseg = int(len(signal / 2))

        frequency, power = scipy.signal.welch(
            signal,
            fs=sampling_rate,
            scaling="density",
            detrend=False,
            nfft=int(nperseg * 2),
            average="mean",
            nperseg=nperseg,
        )

    # Store results
    data = pd.DataFrame({"Frequency": frequency, "Power": power})

    # Filter
    data = data.loc[np.logical_and(data["Frequency"] >= min_frequency, data["Frequency"] <= max_frequency)]

    if show is True:
        ax = data.plot(x="Frequency", y="Power", logy=True, title="Power Spectral Density (PSD)")
        ax.set(xlabel="Frequency (Hz)", ylabel="Spectrum")
        return ax
    else:
        return data

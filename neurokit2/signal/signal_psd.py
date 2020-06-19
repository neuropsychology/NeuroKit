# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal


def signal_psd(
    signal, sampling_rate=1000, method="welch", show=True, min_frequency=0, max_frequency=np.inf, window=None
):
    """Compute the Power Spectral Density (PSD).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    show : bool
        If True, will return a plot. If False, will return the density values that can be plotted externally.
    method : str
        Either 'multitapers' (default; requires the 'mne' package), or 'welch' (requires the 'scipy' package).
    min_frequency : float
        The minimum frequency.
    max_frequency : float
        The maximum frequency.
    window : int
        Length of each window in seconds (for Welch method).

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
        power, frequency = _signal_psd_multitaper(signal, sampling_rate=sampling_rate, min_frequency=min_frequency, max_frequency=max_frequency)

    # BURG
    elif method.lower() in ["burg", "pburg", "spectrum"]:
        raise ValueError("NeuroKit warning: signal_psd(): the 'BURG' method has not been yet implemented.")


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
                "Neurokit warning: signal_psd(): The duration of recording is too short to support a "
                "sufficiently long window for high frequency resolution. Consider using a longer recording "
                "or increasing the `min_frequency`"
                )
            nperseg = int(len(signal / 2))

        # Welch (Scipy)
        if method.lower() in ["welch"]:
            frequency, power = _signal_psd_welch(
    signal, sampling_rate=sampling_rate, nperseg=nperseg
)
        # Lomblombscargle (Scipy)
        elif method.lower() in ["lombscargle", "lomb"]:
            frequency, power = _signal_psd_lomb(
    signal, sampling_rate=sampling_rate, nperseg=nperseg, min_frequency=min_frequency, max_frequency=max_frequency
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


# =============================================================================
# Multitaper method
# =============================================================================
def _signal_psd_multitaper(
    signal, sampling_rate=1000, min_frequency=0, max_frequency=np.inf
):
    try:
        import mne

        power, frequency = mne.time_frequency.psd_array_multitaper(signal,
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
    return power, frequency

# =============================================================================
# Welch method
# =============================================================================
def _signal_psd_welch(
    signal, sampling_rate=1000, nperseg=None
):

    frequency, power = scipy.signal.welch(
        signal,
        fs=sampling_rate,
        scaling="density",
        detrend=False,
        nfft=int(nperseg * 2),
        average="mean",
        nperseg=nperseg,
    )
    return frequency, power


# =============================================================================
# Lomb method
# =============================================================================
def _signal_psd_lomb(
    signal, sampling_rate=1000, nperseg=None, min_frequency=0, max_frequency=np.inf
):

    nfft=int(nperseg * 2)
    if max_frequency == np.inf:
        max_frequency = 10  #sanitize highest frequency

    # Specify frequency range
    frequency = np.linspace(min_frequency, max_frequency, nfft)
    # Compute angular frequencies
    angular_freqs = np.asarray(2 * np.pi / frequency)

    # Specify sample times
    t = np.arange(len(signal))

    power = np.asarray(scipy.signal.lombscargle(t, signal, angular_freqs, normalize=True))

    return frequency, power

# =============================================================================
# Burg method
# =============================================================================
def _signal_psd_burg(signal, order=15, criteria=None, corrected=True):

    # Sanitize order and signal
    if order <= 0.:
        raise ValueError("Order must be > 0")
    if order > len(signal):
        raise ValueError("Order must be less than length signal minus 2")
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    N = len(signal)

    # Initialisation
    # rho is variance of driving white noise process
    rho = sum(abs(signal)**2.) / float(N)
    den = rho * 2. * N

    a = np.zeros(0, dtype=complex)
    ref = np.zeros(0, dtype=complex)
    ef = signal.astype(complex)
    eb = signal.astype(complex)
    temp = 1.

    # Main recursion
    residue_list = np.zeros(order + 1)

    for k in range(0, order):

        # calculate the next order reflection coefficient
        num = sum([ef[j]*eb[j - 1].conjugate() for j in range(k + 1, N)])
        den = temp * den - abs(ef[k])**2 - abs(eb[N - 1])**2
        kp = -2. * num / den

        temp = 1. - abs(kp)**2.
        new_rho = temp * rho

        if criteria is not None:
            # k=k+1 because order goes from 1 to P whereas k starts at 0.
            residue_new = _criteria(criteria=criteria, N=N, k=k+1, rho=new_rho, corrected=corrected)
            if k == 0:
                residue_old = 2. * residue_new

            # Stop as criteria has reached
            if residue_new > residue_old:
                break

        # This should be after the criteria
        residue_old = residue_new
        rho = new_rho
        if rho <= 0:
            raise ValueError("Found a negative value (expected positive strictly) %s. Decrease the order" % rho)

        a.resize(a.size + 1)
        a[k] = kp
        if k == 0:
            for j in range(N-1, k, -1):
                save2 = ef[j]
                ef[j] = save2 + kp * eb[j-1]  # Eq. (8.7)
                eb[j] = eb[j-1] + kp.conjugate() * save2

        else:
            # update the AR coeff
            khalf = (k + 1) // 2  # khalf must be an integer
            for j in range(0, khalf):
                ap = a[j] # previous value
                a[j] = ap + kp * a[k-j-1].conjugate()  # Eq. (8.2)
                if j != k-j-1:
                    a[k-j-1] = a[k-j-1] + kp * ap.conjugate()  # Eq. (8.2)

            # update the prediction error
            for j in range(N-1, k, -1):
                save2 = ef[j]
                ef[j] = save2 + kp * eb[j-1]   # Eq. (8.7)
                eb[j] = eb[j-1] + kp.conjugate() * save2

        # save the reflection coefficient
        ref.resize(ref.size+1)
        ref[k] = kp

    return a, rho, ref

# =============================================================================
# Utilities
# =============================================================================
def _criteria(criteria=None, N=None, k=None, rho=None, corrected=True):
    """criteria to automatically select order in parametric PSD
    AIC, AICc, KIC and AKICc are based on information theory. They attempt to balance the complexity
    (or length) of the model against how well the model fits the data.
    AIC and KIC are biased estimates of the asymmetric and the symmetric Kullback-Leibler divergence
    respectively.
    AICc and AKICc attempt to correct the bias.\

    Parameters
    ----------
    criteria : str
        The criteria to be used.
    N : int
        The sample size of the signal
    k : list, array
        The list of AR order.
    rho : list, array
        The list of rho at order k.
    """
    if criteria == "AIC":
        if corrected is True:
            residue = np.log(rho) + 2. * (k + 1) / (N - k - 2)
        else:
            residue = N * np.log(np.array(rho)) + 2.* (np.array(k) + 1)

    elif criteria == "KIC":
        if corrected is True:
            residue = np.log(rho) + k/N/(N-k) + (3. - (k + 2.) / N) * (k + 1.) / (N - k - 2.)
        else:
            residue = np.log(rho) + 3. * (k + 1.) /float(N)

    return residue

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal


def signal_psd(
    signal, sampling_rate=1000, method="welch", show=True, min_frequency=0, max_frequency=np.inf, window=None, ar_order=15, order_criteria="KIC", order_corrected=True, burg_norm=True
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
    >>> fig3 = nk.signal_psd(signal, method="burg", min_frequency=1)
    >>>
    >>> data = nk.signal_psd(signal, method="multitapers", max_frequency=30, show=False)
    >>> fig4 = data.plot(x="Frequency", y="Power")
    >>> fig4 #doctest: +SKIP
    >>> data = nk.signal_psd(signal, method="welch", max_frequency=30, show=False, min_frequency=1)
    >>> fig5 = data.plot(x="Frequency", y="Power")
    >>> fig5 #doctest: +SKIP

    """
    # Constant Detrend
    signal = signal - np.mean(signal)

    # MNE
    if method.lower() in ["multitapers", "mne"]:
        frequency, power = _signal_psd_multitaper(signal, sampling_rate=sampling_rate, min_frequency=min_frequency, max_frequency=max_frequency)

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
            nperseg = int(len(signal) / 2)

        # Welch (Scipy)
        if method.lower() in ["welch"]:
            frequency, power = _signal_psd_welch(
    signal, sampling_rate=sampling_rate, nperseg=nperseg
)
        # Lombscargle (Scipy)
        elif method.lower() in ["lombscargle", "lomb"]:
            frequency, power = _signal_psd_lomb(
    signal, sampling_rate=sampling_rate, nperseg=nperseg, min_frequency=min_frequency, max_frequency=max_frequency
)

        # BURG
        elif method.lower() in ["burg", "pburg", "spectrum"]:
            frequency, power = _signal_psd_burg(signal, sampling_rate=sampling_rate, order=ar_order, criteria=order_criteria, corrected=order_corrected, side="one-sided", norm=burg_norm, nperseg=nperseg)

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
    return frequency, power

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
        max_frequency = 20  #sanitize highest frequency

    # Specify frequency range
    frequency = np.linspace(min_frequency, max_frequency, nfft)
    # Compute angular frequencies
    #angular_freqs = np.asarray(2 * np.pi / frequency)

    # Specify sample times
    t = np.arange(len(signal))

    power = np.asarray(scipy.signal.lombscargle(t, signal, frequency, normalize=True))

    return frequency, power

# =============================================================================
# Burg method
# =============================================================================


def _signal_psd_burg(signal, sampling_rate=1000, order=15, criteria="KIC", corrected=True, side="one-sided", norm=True, nperseg=None):

    nfft = int(nperseg * 2)
    ar, rho, ref = _signal_arma_burg(signal, order=order, criteria=criteria, corrected=corrected, side=side, norm=norm)
    psd = _signal_psd_from_arma(ar=ar, rho=rho, sampling_rate=sampling_rate, nfft=nfft, side=side, norm=norm)

    # signal is real, not complex
    if nfft % 2 == 0:
        power  = psd[0:int(nfft / 2 + 1)] * 2
    else:
        power  = psd[0:int((nfft + 1) / 2)] * 2

    # angular frequencies, w
    # for one-sided psd, w spans [0, pi]
    # for two-sdied psd, w spans [0, 2pi)
    # for dc-centered psd, w spans (-pi, pi] for even nfft, (-pi, pi) for add nfft
    if side == "one-sided":
        w = np.pi * np.linspace(0, 1, len(power))
#    elif side == "two-sided":
#        w = np.pi * np.linspace(0, 2, len(power), endpoint=False)  #exclude last point
#    elif side == "centerdc":
#        if nfft % 2 == 0:
#            w = np.pi * np.linspace(-1, 1, len(power))
#        else:
#            w = np.pi * np.linspace(-1, 1, len(power) + 1, endpoint=False)  # exclude last point
#            w = w[1:]  # exclude first point (extra)

    frequency = (w * sampling_rate) / (2 * np.pi)

    return frequency, power



def _signal_arma_burg(signal, order=15, criteria="KIC", corrected=True, side="one-sided", norm=True):

    # Sanitize order and signal
    if order <= 0.:
        raise ValueError("Order must be > 0")
    if order > len(signal):
        raise ValueError("Order must be less than length signal minus 2")
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    N = len(signal)

    # Initialisation
    # rho is variance of driving white noise process (prediction error)
    rho = sum(abs(signal)**2.) / float(N)
    denominator = rho * 2. * N

    ar = np.zeros(0, dtype=complex)  #AR parametric signal model estimate
    ref = np.zeros(0, dtype=complex)  #vector K of reflection coefficients (parcor coefficients)
    ef = signal.astype(complex)  #forward prediction error
    eb = signal.astype(complex)  #backward prediction error
    temp = 1.

    # Main recursion

    for k in range(0, order):

        # calculate the next order reflection coefficient
        numerator = sum([ef[j]*eb[j - 1].conjugate() for j in range(k + 1, N)])
        denominator = temp * denominator - abs(ef[k])**2 - abs(eb[N - 1])**2
        kp = -2. * numerator / denominator

        # Update the prediction error
        temp = 1. - abs(kp)**2.
        new_rho = temp * rho

        if criteria is not None:
            # k=k+1 because order goes from 1 to P whereas k starts at 0.
            residual_new = _criteria(criteria=criteria, N=N, k=k+1, rho=new_rho, corrected=corrected)
            if k == 0:
                residual_old = 2. * abs(residual_new)

            # Stop as criteria has reached
            if residual_new > residual_old:
                break

        # This should be after the criteria
        residual_old = residual_new
        rho = new_rho
        if rho <= 0:
            raise ValueError("Found a negative value (expected positive strictly) %s."
                             "Decrease the order" % rho)

        ar = np.resize(ar, ar.size + 1)
        ar[k] = kp
        if k == 0:
            for j in range(N-1, k, -1):
                ef_previous = ef[j]  # previous value
                ef[j] = ef_previous + kp * eb[j-1]  # Eq. (8.7)
                eb[j] = eb[j-1] + kp.conjugate() * ef_previous

        else:
            # Update the AR coeff
            khalf = (k + 1) // 2  # khalf must be an integer
            for j in range(0, khalf):
                ar_previous = ar[j]  # previous value
                ar[j] = ar_previous + kp * ar[k-j-1].conjugate()  # Eq. (8.2)
                if j != k-j-1:
                    ar[k-j-1] = ar[k-j-1] + kp * ar_previous.conjugate()  # Eq. (8.2)

            # Update the forward and backward prediction errors
            for j in range(N-1, k, -1):
                ef_previous = ef[j]  # previous value
                ef[j] = ef_previous + kp * eb[j-1]   # Eq. (8.7)
                eb[j] = eb[j-1] + kp.conjugate() * ef_previous

        # save the reflection coefficient
        ref.resize(ref.size+1)
        ref[k] = kp

    return ar, rho, ref



# =============================================================================
# Utilities
# =============================================================================


def _criteria(criteria=None, N=None, k=None, rho=None, corrected=True):
    """Criteria to automatically select order in parametric PSD.

    AIC, AICc, KIC and AKICc are based on information theory. They attempt to balance the complexity
    (or length) of the model against how well the model fits the data.
    AIC and KIC are biased estimates of the asymmetric and the symmetric Kullback-Leibler divergence
    respectively. AICc and AKICc attempt to correct the bias.

    Parameters
    ----------
    criteria : str
        The criteria to be used. The critera can be one of the following: AIC (Akaike Information Criterion),
        KIC (Kullback Iinformation Criterion), FPE (Final Prediction Error Criterion), MDL (Minimum
        Description Length), CAT (Criterion Autoregressive Transfer Function), AIC order-selection using
        eigen values, MDL order-selection using eigen values.
    N : int
        The sample size of the signal.
    k : int
        The AR order.
    rho : int
        The rho at order k.
    """
    if criteria == "AIC":
        if corrected is True:
            residual = np.log(rho) + 2. * (k + 1) / (N - k - 2)
        else:
            residual = N * np.log(np.array(rho)) + 2. * (np.array(k) + 1)

    elif criteria == "KIC":
        if corrected is True:
            residual = np.log(rho) + k/N/(N-k) + (3. - (k + 2.) / N) * (k + 1.) / (N - k - 2.)
        else:
            residual = np.log(rho) + 3. * (k + 1.) / float(N)

    elif criteria == "FPE":
        fpe = rho * (N + k + 1.) / (N- k -1)
        return fpe

    elif criteria == "MDL":
        mdl = N * np.log(rho) + k * np.log(N)
        return mdl

    return residual


def _signal_psd_from_arma(ar=None, ma=None, rho=1., sampling_rate=1000, nfft=None, side="one-sided", norm=False):

    if ar is None and ma is None:
        raise ValueError("Either AR or MA model must be provided")

    psd = np.zeros(nfft, dtype=complex)

    if ar is not None:
        ip = len(ar)
        den = np.zeros(nfft, dtype=complex)
        den[0] = 1.+0j
        for k in range(0, ip):
            den[k+1] = ar[k]
        denf = np.fft.fft(den, nfft)

    if ma is not None:
        iq = len(ma)
        num = np.zeros(nfft, dtype=complex)
        num[0] = 1.+0j
        for k in range(0, iq):
            num[k+1] = ma[k]
        numf = np.fft(num, nfft)

    if ar is not None and ma is not None:
        psd = rho / sampling_rate * abs(numf)**2. / abs(denf)**2.
    elif ar is not None:
        psd = rho / sampling_rate / abs(denf)**2.
    elif ma is not None:
        psd = rho / sampling_rate * abs(numf)**2.


    psd = np.real(psd)  # The PSD is a twosided PSD.

    # convert to one-sided
    if side == "one-sided":
        assert len(psd) % 2 == 0
        one_side_psd = np.array(psd[0:len(psd)//2 + 1]) * 2.
        one_side_psd[0] /= 2.
#        one_side_psd[-1] = psd[-1]
        psd = one_side_psd

    # convert to centerdc
    elif side == "centerdc":
        first_half = psd[0:len(psd)//2]
        second_half = psd[len(psd)//2:]
        rotate_second_half = (second_half[-1:] + second_half[:-1])
        center_psd = np.concatenate((rotate_second_half, first_half))
        center_psd[0] = psd[-1]
        psd = center_psd

    if norm == True:
        psd /= max(psd)

    return psd

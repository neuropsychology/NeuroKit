from warnings import warn

import numpy as np
import pandas as pd
import scipy.signal

from ..misc import NeuroKitWarning


def signal_psd(
    signal,
    sampling_rate=1000,
    method="welch",
    show=False,
    normalize=True,
    min_frequency="default",
    max_frequency=np.inf,
    window=None,
    window_type="hann",
    order=16,
    order_criteria="KIC",
    order_corrected=True,
    silent=True,
    t=None,
    **kwargs,
):
    """**Compute the Power Spectral Density (PSD)**

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    method : str
        Either ``"welch"`` (default), ``"fft"``, ``"multitapers"`` (requires the 'mne' package),
        ``"lombscargle"`` (requires the 'astropy' package) or ``"burg"``.
    show : bool
        If ``True``, will return a plot. If ``False``, will return the density values that can be
        plotted externally.
    normalize : bool
        Normalization of power by maximum PSD value. Default to ``True``.
        Normalization allows comparison between different PSD methods.
    min_frequency : str, float
        The minimum frequency. If default, min_frequency is chosen based on the sampling rate and
        length of signal to optimize the frequency resolution.
    max_frequency : float
        The maximum frequency.
    window : int
        Length of each window in seconds (for "Welch" method). If ``None`` (default), window will be
        automatically calculated to capture at least 2 cycles of min_frequency. If the length of
        recording does not allow the formal, window will be default to half of the length of
        recording.
    window_type : str
        Desired window to use. Defaults to ``"hann"``. See :func:`.scipy.signal.get_window` for list
        of windows.
    order : int
        The order of autoregression (only used for autoregressive (AR) methods such as ``"burg"``).
    order_criteria : str
        The criteria to automatically select order in parametric PSD (only used for autoregressive
        (AR) methods such as ``"burg"``).
    order_corrected : bool
        Should the order criteria (AIC or KIC) be corrected? If unsure which method to use to choose
        the order, rely on the default (i.e., the corrected KIC).
    silent : bool
        If ``False``, warnings will be printed. Default to ``True``.
    t : array
        The timestamps corresponding to each sample in the signal, in seconds
        (for ``"lombscargle"`` method). Defaults to None.
    **kwargs : optional
        Keyword arguments to be passed to :func:`.scipy.signal.welch` or :func:`.np.fft.rfft` (when
        method is 'fft', such as `n`, which determines the number of windows).

    See Also
    --------
    signal_filter, mne.time_frequency.psd_array_multitaper, scipy.signal.welch

    Returns
    -------
    data : pd.DataFrame
        A DataFrame containing the Power Spectrum values and a plot if
        ``show`` is ``True``.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=[5, 6, 50, 52, 80], noise=0.5)

      # FFT method (based on numpy)
      @savefig p_signal_psd1.png scale=100%
      psd_multitapers = nk.signal_psd(signal, method="fft", show=True)
      @suppress
      plt.close

    .. ipython:: python

      # Welch method (based on scipy)
      @savefig p_signal_psd2.png scale=100%
      psd_welch = nk.signal_psd(signal, method="welch", min_frequency=1, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      # Multitapers method (requires MNE)
      @savefig p_signal_psd3.png scale=100%
      psd_multitapers = nk.signal_psd(signal, method="multitapers", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      # Burg method
      @savefig p_signal_psd4.png scale=100%
      psd_burg = nk.signal_psd(signal, method="burg", min_frequency=1, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      # Lomb method (requires AstroPy)
      @savefig p_signal_psd5.png scale=100%
      psd_lomb = nk.signal_psd(signal, method="lomb", min_frequency=1, show=True)
      @suppress
      plt.close()

    """
    # Sanitize signal
    if isinstance(signal, pd.Series):
        signal = signal.values
    signal = np.asarray(signal, dtype=float)

    # Constant Detrend
    signal = signal - np.mean(signal)

    # Sanitize method name
    method = method.lower()

    # Sanitize min_frequency
    N = len(signal)
    if isinstance(min_frequency, str):
        if sampling_rate is None:
            if t is None or len(t) < 2:
                raise ValueError("NeuroKit error: signal_psd(): 't' must be provided when 'sampling_rate' is None.")
            # This is to compute min_frequency if both min_frequency and sampling_rate are not provided (#800)
            min_frequency = (2 / np.median(np.diff(t))) / (N / 2)  # for high frequency resolution
        else:
            min_frequency = (2 * sampling_rate) / (N / 2)  # for high frequency resolution

    # MNE
    if method in ["multitaper", "multitapers", "mne"]:
        frequency, power = _signal_psd_multitaper(
            signal,
            sampling_rate=sampling_rate,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
        )

    # FFT (Numpy)
    elif method in ["fft"]:
        frequency, power = _signal_psd_fft(signal, sampling_rate=sampling_rate, **kwargs)

    # Lombscargle (AstroPy)
    elif method in ["lombscargle", "lomb"]:
        frequency, power = _signal_psd_lomb(
            signal,
            sampling_rate=sampling_rate,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            t=t,
        )

    # Methods that are using a window
    elif method in ["welch", "burg", "pburg", "spectrum"]:
        # Define window length
        if min_frequency == 0:
            min_frequency = 0.001  # sanitize min_frequency

        if window is not None:
            nperseg = int(window * sampling_rate)
        else:
            # to capture at least 2 cycles of min_frequency
            nperseg = int((2 / min_frequency) * sampling_rate)

        # in case duration of recording is not sufficient
        if nperseg > N / 2:
            if silent is False:
                warn(
                    "The duration of recording is too short to support a"
                    " sufficiently long window for high frequency resolution."
                    " Consider using a longer recording or increasing the `min_frequency`",
                    category=NeuroKitWarning,
                )
            nperseg = int(N / 2)

        # Welch (Scipy)
        if method in ["welch"]:
            frequency, power = _signal_psd_welch(
                signal,
                sampling_rate=sampling_rate,
                nperseg=nperseg,
                window_type=window_type,
                **kwargs,
            )

        # BURG
        elif method in ["burg", "pburg", "spectrum"]:
            frequency, power = _signal_psd_burg(
                signal,
                sampling_rate=sampling_rate,
                order=order,
                criteria=order_criteria,
                corrected=order_corrected,
                side="one-sided",
                nperseg=nperseg,
            )

    else:
        raise ValueError(
            f"NeuroKit error: signal_psd(): 'method' '{method}' not recognized. "
            "Must be one of 'welch', 'fft', 'multitapers', 'lombscargle', or 'burg'."
        )

    # Normalize
    if normalize is True and len(power) > 0:
        max_power = np.max(power)
        if max_power > 0:
            power = power / max_power

    # Store results
    data = pd.DataFrame({"Frequency": frequency, "Power": power})

    # Filter
    data = data.loc[np.logical_and(data["Frequency"] >= min_frequency, data["Frequency"] <= max_frequency)]
    #    data["Power"] = 10 * np.log(data["Power"])

    if show is True:
        ax = data.plot(
            x="Frequency",
            y="Power",
            title="Power Spectral Density (" + str(method) + " method)",
        )
        ax.set(xlabel="Frequency (Hz)", ylabel="Spectrum")

    return data


# =============================================================================
# Multitaper method
# =============================================================================


def _signal_psd_fft(signal, sampling_rate=1000, n=None):
    # Power-spectrum density (PSD)
    power = np.abs(np.fft.rfft(signal, n=n)) ** 2
    n_fft = len(signal) if n is None else n
    frequency = np.fft.rfftfreq(n_fft, d=1 / sampling_rate)
    return frequency, power


# =============================================================================
# Multitaper method
# =============================================================================


def _signal_psd_multitaper(signal, sampling_rate=1000, min_frequency=0, max_frequency=np.inf):
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "NeuroKit error: signal_psd(): the 'mne'",
            " module is required for the 'mne' method to run.",
            " Please install it first (`pip install mne`).",
        ) from e

    power, frequency = mne.time_frequency.psd_array_multitaper(
        signal,
        sfreq=sampling_rate,
        fmin=min_frequency,
        fmax=max_frequency,
        adaptive=True,
        normalization="full",
        verbose=False,
    )

    return frequency, power


# =============================================================================
# Welch method
# =============================================================================


def _signal_psd_welch(signal, sampling_rate=1000, nperseg=None, window_type="hann", **kwargs):
    if nperseg is not None:
        nfft = int(nperseg * 2)
    else:
        nfft = None

    frequency, power = scipy.signal.welch(
        signal,
        fs=sampling_rate,
        scaling="density",
        detrend=False,
        nfft=nfft,
        average="mean",
        nperseg=nperseg,
        window=window_type,
        **kwargs,
    )

    return frequency, power


# =============================================================================
# Lomb method
# =============================================================================


def _signal_psd_lomb(signal, sampling_rate=1000, min_frequency=0, max_frequency=np.inf, t=None):
    try:
        import astropy.timeseries

        if t is None:
            if max_frequency == np.inf:
                max_frequency = sampling_rate / 2  # sanitize highest frequency
            t = np.arange(len(signal)) / sampling_rate
            frequency, power = astropy.timeseries.LombScargle(t, signal, normalization="psd").autopower(
                minimum_frequency=min_frequency, maximum_frequency=max_frequency
            )
        else:
            # determine maximum frequency with astropy defaults for unevenly spaced data
            # https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.autopower
            frequency, power = astropy.timeseries.LombScargle(t, signal, normalization="psd").autopower(
                minimum_frequency=min_frequency
            )

    except ImportError as e:
        raise ImportError(
            "NeuroKit error: signal_psd(): the 'astropy'",
            " module is required for the 'lomb' method to run.",
            " Please install it first (`pip install astropy`).",
        ) from e

    return frequency, power


# =============================================================================
# Burg method
# =============================================================================


def _signal_psd_burg(
    signal,
    sampling_rate=1000,
    order=16,
    criteria="KIC",
    corrected=True,
    side="one-sided",
    nperseg=None,
):
    if nperseg is None:
        nperseg = max(1, len(signal) // 2)
    nfft = max(2, int(nperseg * 2))
    ar, rho, _ = _signal_arma_burg(signal, order=order, criteria=criteria, corrected=corrected)
    power = _signal_psd_from_arma(ar=ar, rho=rho, sampling_rate=sampling_rate, nfft=nfft, side=side)

    # angular frequencies, w
    # for one-sided psd, w spans [0, pi] for even nfft
    # for two-sided psd, w spans [0, 2pi)
    # for dc-centered psd, w spans [-pi, pi)
    if side == "one-sided":
        w = np.pi * np.linspace(0, 1, len(power))
    elif side == "two-sided":
        w = np.pi * np.linspace(0, 2, len(power), endpoint=False)
    elif side == "centerdc":
        w = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nfft))
    else:
        raise ValueError(f"NeuroKit error: _signal_psd_burg(): unknown 'side' '{side}'.")

    frequency = (w * sampling_rate) / (2 * np.pi)

    return frequency, power


def _signal_arma_burg(signal, order=16, criteria="KIC", corrected=True):
    # Sanitize order and signal
    N = len(signal)
    if order <= 0.0:
        raise ValueError("Order must be > 0")
    if order > N:
        raise ValueError("Order must be less than length signal minus 2")
    signal = np.asarray(signal)

    # Initialisation
    # rho is variance of driving white noise process (prediction error)
    rho = float(np.vdot(signal, signal).real) / float(N)
    if rho == 0.0:
        return np.zeros(1, dtype=complex), 0.0, np.zeros(1, dtype=complex)
    denominator = rho * 2.0 * N

    ar = np.zeros(order, dtype=complex)  # AR parametric signal model estimate
    ref = np.zeros(order, dtype=complex)  # vector K of reflection coefficients (parcor coefficients)
    ef = signal.astype(complex)  # forward prediction error
    eb = signal.astype(complex)  # backward prediction error
    temp = 1.0
    k_used = 0

    # Main recursion
    for k in range(0, order):
        if k >= N - 1:
            break
        # calculate the next order reflection coefficient
        numerator = np.vdot(eb[k : N - 1], ef[k + 1 : N])
        denominator = temp * denominator - abs(ef[k]) ** 2 - abs(eb[N - 1]) ** 2
        if denominator <= 0:
            break
        kp = -2.0 * numerator / denominator

        # Update the prediction error
        temp = 1.0 - abs(kp) ** 2.0
        new_rho = temp * rho

        if criteria is not None:
            # k=k+1 because order goes from 1 to P whereas k starts at 0.
            residual_new = _criteria(criteria=criteria, N=N, k=k + 1, rho=new_rho, corrected=corrected)
            if k == 0:
                residual_old = 2.0 * abs(residual_new)

            # Stop as criteria has reached
            if residual_new > residual_old:
                break

            # This should be after the criteria
            residual_old = residual_new
        rho = new_rho
        if rho <= 0:
            raise ValueError(f"Found a negative value (expected positive strictly) {rho}. Decrease the order.")

        # Update the AR coeff (Levinson-Durbin recursion)
        if k > 0:
            ar_prev = ar[:k].copy()
            ar[:k] = ar_prev + kp * ar_prev[::-1].conjugate()
        ar[k] = kp

        # Update the forward and backward prediction errors
        ef_prev = ef[k + 1 : N].copy()
        eb_prev = eb[k : N - 1].copy()
        ef[k + 1 : N] = ef_prev + kp * eb_prev
        eb[k + 1 : N] = eb_prev + kp.conjugate() * ef_prev

        # save the reflection coefficient
        ref[k] = kp
        k_used = k + 1

    return ar[:k_used], rho, ref[:k_used]


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
    corrected : bool
        Specify for AIC and KIC methods.

    Returns
    -------
    residual : Union[int, float]
        Residuals to select the optimal order.

    """
    if criteria == "AIC":
        if corrected is True:
            if N - k - 2 <= 0:
                return np.inf
            residual = np.log(rho) + 2.0 * (k + 1) / (N - k - 2)
        else:
            residual = N * np.log(np.array(rho)) + 2.0 * (np.array(k) + 1)

    elif criteria == "KIC":
        if corrected is True:
            if N - k - 2 <= 0 or N - k <= 0:
                return np.inf
            residual = np.log(rho) + k / N / (N - k) + (3.0 - (k + 2.0) / N) * (k + 1.0) / (N - k - 2.0)
        else:
            residual = np.log(rho) + 3.0 * (k + 1.0) / float(N)

    elif criteria == "FPE":
        if N - k - 1 <= 0:
            return np.inf
        return rho * (N + k + 1.0) / (N - k - 1)

    elif criteria == "MDL":
        return N * np.log(rho) + k * np.log(N)

    else:
        raise ValueError(f"NeuroKit error: _criteria(): unknown criteria '{criteria}'.")

    return residual


def _signal_psd_from_arma(ar=None, ma=None, rho=1.0, sampling_rate=1000, nfft=None, side="one-sided"):
    if ar is None and ma is None:
        raise ValueError("Either AR or MA model must be provided")

    psd = np.zeros(nfft, dtype=complex)

    if ar is not None:
        den = np.zeros(max(nfft, len(ar) + 1), dtype=complex)
        den[0] = 1.0 + 0j
        den[1 : len(ar) + 1] = ar
        denf = np.fft.fft(den, nfft)

    if ma is not None:
        num = np.zeros(max(nfft, len(ma) + 1), dtype=complex)
        num[0] = 1.0 + 0j
        num[1 : len(ma) + 1] = ma
        numf = np.fft.fft(num, nfft)

    if ar is not None and ma is not None:
        psd = rho / sampling_rate * abs(numf) ** 2.0 / abs(denf) ** 2.0
    elif ar is not None:
        psd = rho / sampling_rate / abs(denf) ** 2.0
    elif ma is not None:
        psd = rho / sampling_rate * abs(numf) ** 2.0

    psd = np.real(psd)  # The PSD is a twosided PSD.

    # convert to one-sided
    if side == "one-sided":
        if len(psd) % 2 == 0:
            one_side_psd = np.array(psd[0 : len(psd) // 2 + 1]) * 2.0
            one_side_psd[0] /= 2.0
            one_side_psd[-1] /= 2.0
        else:
            one_side_psd = np.array(psd[0 : (len(psd) + 1) // 2]) * 2.0
            one_side_psd[0] /= 2.0
        psd = one_side_psd

    # convert to centerdc
    elif side == "centerdc":
        psd = np.fft.fftshift(psd)

    return psd

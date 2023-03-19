import numpy as np

from ..misc import check_random_state


def signal_surrogate(signal, method="IAAFT", random_state=None, **kwargs):
    """**Create Signal Surrogates**

    Generate a surrogate version of a signal. Different methods are available, such as:

    * **random**: Performs a random permutation of the signal value. This way, the signal
      distribution is unaffected and the serial correlations are cancelled, yielding a whitened
      signal with an distribution identical to that of the original.
    * **IAAFT**: Returns an Iterative Amplitude Adjusted Fourier Transform (IAAFT) surrogate.
      It is a phase randomized, amplitude adjusted surrogates that have the same power spectrum
      (to a very high accuracy) and distribution as the original data, using an iterative scheme.


    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        Can be ``"random"`` or ``"IAAFT"``.
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator. See for ``misc.check_random_state`` for further information.
    **kwargs
        Other keywords arguments, such as ``max_iter`` (by default 1000).

    Returns
    -------
    surrogate : array
        Surrogate signal.

    Examples
    --------
    Create surrogates using different methods.

    .. ipython:: python

      import neurokit2 as nk
      import matplotlib.pyplot as plt

      signal = nk.signal_simulate(duration = 1, frequency = [3, 5], noise = 0.1)
      surrogate_iaaft = nk.signal_surrogate(signal, method = "IAAFT")
      surrogate_random = nk.signal_surrogate(signal, method = "random")

      @savefig p_signal_surrogate1.png scale=100%
      plt.plot(surrogate_random, label = "Random Surrogate")
      plt.plot(surrogate_iaaft, label = "IAAFT Surrogate")
      plt.plot(signal, label = "Original")
      plt.legend()
      @suppress
      plt.close()

    As we can see, the signal pattern is destroyed by random surrogates, but not in the IAAFT one.
    And their distributions are identical:

    .. ipython:: python

      @savefig p_signal_surrogate2.png scale=100%
      plt.plot(*nk.density(signal), label = "Original")
      plt.plot(*nk.density(surrogate_iaaft), label = "IAAFT Surrogate")
      plt.plot(*nk.density(surrogate_random), label = "Random Surrogate")
      plt.legend()
      @suppress
      plt.close()

    However, the power spectrum of the IAAFT surrogate is preserved.

    .. ipython:: python

      f = nk.signal_psd(signal, max_frequency=20)
      f["IAAFT"] = nk.signal_psd(surrogate_iaaft, max_frequency=20)["Power"]
      f["Random"] = nk.signal_psd(surrogate_random, max_frequency=20)["Power"]
      @savefig p_signal_surrogate3.png scale=100%
      f.plot("Frequency", ["Power", "IAAFT", "Random"])
      @suppress
      plt.close()


    References
    ----------
    * Schreiber, T., & Schmitz, A. (1996). Improved surrogate data for nonlinearity tests. Physical
      review letters, 77(4), 635.

    """
    # TODO: when discrete signal is detected, run surrogate of markov chains
    # https://github.com/Frederic-vW/eeg_microstates/blob/eeg_microstates3.py#L861
    # Or markov_simulate()

    # Seed the random generator for reproducible results
    rng = check_random_state(random_state)

    method = method.lower()
    if method == "random":
        surrogate = rng.permutation(signal)
    elif method == "iaaft":
        surrogate, _, _ = _signal_surrogate_iaaft(signal, rng=rng, **kwargs)

    return surrogate


def _signal_surrogate_iaaft(signal, max_iter=1000, atol=1e-8, rtol=1e-10, rng=None):
    """IAAFT
    max_iter : int
        Maximum iterations to be performed while checking for convergence. Convergence can be
        achieved before maximum interation.
    atol : float
        Absolute tolerance for checking convergence.
    rtol : float
        Relative tolerance for checking convergence. If both atol and rtol are set to zero, the
        iterations end only when the RMSD stops changing or when maximum iteration is reached.

    Returns
    -------
    surrogate : array
        Surrogate series with (almost) the same power spectrum and distribution.
    i : int
        Number of iterations that have been performed.
    rmsd : float
        Root-mean-square deviation (RMSD) between the absolute squares of the Fourier amplitudes of
        the surrogate series and that of the original series.

    """

    # Calculate "true" Fourier amplitudes and sort the series
    amplitudes = np.abs(np.fft.rfft(signal))
    sort = np.sort(signal)

    # Previous and current error
    previous_error, current_error = (-1, 1)

    # Start with a random permutation
    t = np.fft.rfft(rng.permutation(signal))

    for i in range(max_iter):
        # Match power spectrum
        s = np.real(np.fft.irfft(amplitudes * t / np.abs(t), n=len(signal)))

        # Match distribution by rank ordering
        surrogate = sort[np.argsort(np.argsort(s))]

        t = np.fft.rfft(surrogate)
        current_error = np.sqrt(np.mean((amplitudes ** 2 - np.abs(t) ** 2) ** 2))

        # Check convergence
        if abs(current_error - previous_error) <= atol + rtol * abs(previous_error):
            break
        previous_error = current_error

    # Normalize error w.r.t. mean of the "true" power spectrum.
    rmsd = current_error / np.mean(amplitudes ** 2)
    return surrogate, i, rmsd

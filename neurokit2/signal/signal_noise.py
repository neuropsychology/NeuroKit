import numpy as np

from ..misc import check_random_state


def signal_noise(duration=10, sampling_rate=1000, beta=1, random_state=None):
    """**Simulate noise**

    This function generates pure Gaussian ``(1/f)**beta`` noise. The power-spectrum of the generated
    noise is proportional to ``S(f) = (1 / f)**beta``. The following categories of noise have been
    described:

    * violet noise: beta = -2
    * blue noise: beta = -1
    * white noise: beta = 0
    * flicker / pink noise: beta = 1
    * brown noise: beta = 2

    Parameters
    ----------
    duration : float
        Desired length of duration (s).
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    beta : float
        The noise exponent.
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator. See for ``misc.check_random_state`` for further information.

    Returns
    -------
    noise : array
        The signal of pure noise.

    References
    ----------
    * Timmer, J., & Koenig, M. (1995). On generating power law noise. Astronomy and Astrophysics,
      300, 707.
    * https://github.com/felixpatzelt/colorednoise
    * https://en.wikipedia.org/wiki/Colors_of_noise

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      import matplotlib.pyplot as plt

      # Generate pure noise
      violet = nk.signal_noise(beta=-2)
      blue = nk.signal_noise(beta=-1)
      white = nk.signal_noise(beta=0)
      pink = nk.signal_noise(beta=1)
      brown = nk.signal_noise(beta=2)

      # Visualize
      @savefig p_signal_noise1.png scale=100%
      nk.signal_plot([violet, blue, white, pink, brown],
                      standardize=True,
                      labels=["Violet", "Blue", "White", "Pink", "Brown"])
      @suppress
      plt.close()

    .. ipython:: python

      # Visualize spectrum
      psd_violet = nk.signal_psd(violet, sampling_rate=200, method="fft")
      psd_blue = nk.signal_psd(blue, sampling_rate=200, method="fft")
      psd_white = nk.signal_psd(white, sampling_rate=200, method="fft")
      psd_pink = nk.signal_psd(pink, sampling_rate=200, method="fft")
      psd_brown = nk.signal_psd(brown, sampling_rate=200, method="fft")

      @savefig p_signal_noise2.png scale=100%
      plt.loglog(psd_violet["Frequency"], psd_violet["Power"], c="violet")
      plt.loglog(psd_blue["Frequency"], psd_blue["Power"], c="blue")
      plt.loglog(psd_white["Frequency"], psd_white["Power"], c="grey")
      plt.loglog(psd_pink["Frequency"], psd_pink["Power"], c="pink")
      plt.loglog(psd_brown["Frequency"], psd_brown["Power"], c="brown")
      @suppress
      plt.close()

    """
    # Seed the random generator for reproducible results
    rng = check_random_state(random_state)

    # The number of samples in the time series
    n = int(duration * sampling_rate)

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = np.fft.rfftfreq(n, d=1 / sampling_rate)

    # Build scaling factors for all frequencies
    fmin = 1.0 / n  # Low frequency cutoff
    f[f < fmin] = fmin
    f = f ** (-beta / 2.0)

    # Calculate theoretical output standard deviation from scaling
    w = f[1:].copy()
    w[-1] *= (1 + (n % 2)) / 2.0  # correct f = +-0.5
    sigma = 2 * np.sqrt(np.sum(w ** 2)) / n

    # Generate scaled random power + phase, adjusting size to
    # generate one Fourier component per frequency
    sr = rng.normal(scale=f, size=len(f))
    si = rng.normal(scale=f, size=len(f))

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not n % 2:
        si[..., -1] = 0

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0

    # Combine power + corrected phase to Fourier components
    s = sr + 1j * si

    # Transform to real time series & scale to unit variance
    y = np.fft.irfft(s, n=n) / sigma

    return y

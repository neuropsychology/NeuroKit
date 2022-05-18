# -*- coding: utf-8 -*-
import numpy as np

from .utils_complexity_attractor import _attractor_lorenz


def complexity_simulate(
    duration=10, sampling_rate=1000, method="ornstein", hurst_exponent=0.5, **kwargs
):
    """**Simulate chaotic time series**

    This function generates a chaotic signal using different algorithms and complex systems.

    * **Mackey-Glass:** Generates time series using the discrete approximation of the
      Mackey-Glass delay differential equation described by Grassberger & Procaccia (1983).
    * **Ornstein-Uhlenbeck**
    * **Lorenz**
    * **Random walk**

    Parameters
    ----------
    duration : int
        Desired length of duration (s).
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    duration : int
        The desired length in samples.
    method : str
        The method. can be ``"hurst"`` for a (fractional) Ornstein-Uhlenbeck process, ``"lorenz"``
        for the first dimension of a Lorenz system, ``"mackeyglass"`` to use the Mackey-Glass
        equation, or ``random`` to generate a random-walk.
    hurst_exponent : float
        Defaults to ``0.5``.
    **kwargs
        Other arguments.

    Returns
    -------
    array
        Simulated complexity time series.

    Examples
    ------------
    **Lorenz System**

    .. ipython:: python

      import neurokit2 as nk

      signal = nk.complexity_simulate(duration=5, sampling_rate=1000, method="lorenz")
      @savefig p_complexity_simulate1.png scale=100%
      nk.signal_plot(signal)
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_simulate2.png scale=100%
      nk.complexity_attractor(nk.complexity_embedding(signal, delay = 5), alpha=1, color="blue")
      @suppress
      plt.close()

    **Ornstein System**

    .. ipython:: python

      signal = nk.complexity_simulate(duration=30, sampling_rate=100, method="ornstein")
      @savefig p_complexity_simulate3.png scale=100%
      nk.signal_plot(signal, color = "red")
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_simulate4.png scale=100%
      nk.complexity_attractor(nk.complexity_embedding(signal, delay = 100), alpha=1, color="red")
      @suppress
      plt.close()

    **Mackey-Glass System**

    .. ipython:: python

      signal = nk.complexity_simulate(duration=1, sampling_rate=1000, method="mackeyglass")
      @savefig p_complexity_simulate5.png scale=100%
      nk.signal_plot(signal, color = "green")
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_simulate6.png scale=100%
      nk.complexity_attractor(nk.complexity_embedding(signal, delay = 25), alpha=1, color="green")
      @suppress
      plt.close()

    **Random walk**

    .. ipython:: python

      signal = nk.complexity_simulate(duration=30, sampling_rate=100, method="randomwalk")
      @savefig p_complexity_simulate7.png scale=100%
      nk.signal_plot(signal, color = "orange")
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_simulate8.png scale=100%
      nk.complexity_attractor(nk.complexity_embedding(signal, delay = 100), alpha=1, color="orange")
      @suppress
      plt.close()

    """
    method = method.lower()
    if method in ["fractal", "fractional", "hurst", "ornsteinuhlenbeck", "ornstein"]:
        signal = _complexity_simulate_ornstein(
            duration=duration, sampling_rate=sampling_rate, hurst_exponent=hurst_exponent, **kwargs
        )
    elif method in ["lorenz"]:
        # x-dimension of Lorenz system
        signal = _attractor_lorenz(sampling_rate=sampling_rate, duration=duration, **kwargs)[:, 0]
    elif method in ["mackeyglass"]:
        signal = _complexity_simulate_mackeyglass(
            duration=duration, sampling_rate=sampling_rate, **kwargs
        )
    else:
        signal = _complexity_simulate_randomwalk(int(duration * sampling_rate))
    return signal


# =============================================================================
# Methods
# =============================================================================
def _complexity_simulate_mackeyglass(
    duration=10, sampling_rate=1000, x0="fixed", a=0.2, b=0.1, c=10.0, n=1000, discard=250
):
    """Generate time series using the Mackey-Glass equation. Generates time series using the discrete approximation of
    the Mackey-Glass delay differential equation described by Grassberger & Procaccia (1983).

    Taken from nolitsa (https://github.com/manu-mannattil/nolitsa/blob/master/nolitsa/data.py#L223).

    Parameters
    ----------
    duration : int
        Duration of the time series to be generated.
    sampling_rate : float
        Sampling step of the time series.  It is useful to pick something between tau/100 and tau/10,
        with tau/sampling_rate being a factor of n.  This will make sure that there are only whole
        number indices. Defaults to 1000.
    x0 : array
        Initial condition for the discrete map. Should be of length n. Can be "fixed", "random", or
        a vector of size n.
    a : float
        Constant a in the Mackey-Glass equation. Defaults to 0.2.
    b : float
        Constant b in the Mackey-Glass equation. Defaults to 0.1.
    c : float
        Constant c in the Mackey-Glass equation. Defaults to 10.0
    n : int
        The number of discrete steps into which the interval between t and t + tau should be divided.
        This results in a time step of tau/n and an n + 1 dimensional map. Defaults to 1000.
    discard : int
        Number of n-steps to discard in order to eliminate transients. A total of n*discard steps will
        be discarded. Defaults to 250.

    Returns
    -------
    array
        Simulated complexity time series.

    """
    length = duration * sampling_rate
    tau = sampling_rate / 2 * 100
    sampling_rate = int(n * sampling_rate / tau)
    grids = int(n * discard + sampling_rate * length)
    x = np.zeros(grids)

    if isinstance(x0, str):
        if x0 == "random":
            x[:n] = 0.5 + 0.05 * (-1 + 2 * np.random.random(n))
        else:
            x[:n] = np.ones(n)
    else:
        x[:n] = x0

    A = (2 * n - b * tau) / (2 * n + b * tau)
    B = a * tau / (2 * n + b * tau)

    for i in range(n - 1, grids - 1):
        x[i + 1] = A * x[i] + B * (
            x[i - n] / (1 + x[i - n] ** c) + x[i - n + 1] / (1 + x[i - n + 1] ** c)
        )
    return x[n * discard :: sampling_rate]


def _complexity_simulate_ornstein(
    duration=10, sampling_rate=1000, theta=0.3, sigma=0.1, hurst_exponent=0.7
):
    """This is based on https://github.com/LRydin/MFDFA.

    Parameters
    ----------
    duration : int
        The desired length in samples.
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second). Defaults to 1000Hz.
    theta : float
        Drift. Defaults to 0.3.
    sigma : float
        Diffusion. Defaults to 0.1.
    hurst_exponent : float
        Defaults to 0.7.

    Returns
    -------
    array
        Simulated complexity time series.

    """
    # Time array
    length = duration * sampling_rate

    # The fractional Gaussian noise
    dB = (duration ** hurst_exponent) * _complexity_simulate_fractionalnoise(
        size=length, hurst_exponent=hurst_exponent
    )

    # Initialise the array y
    y = np.zeros([length])

    # Integrate the process
    for i in range(1, length):
        y[i] = y[i - 1] - theta * y[i - 1] * (1 / sampling_rate) + sigma * dB[i]
    return y


def _complexity_simulate_fractionalnoise(size=1000, hurst_exponent=0.5):
    """Generates fractional Gaussian noise.

    Generates fractional Gaussian noise with a Hurst index H in (0,1). If H = 1/2 this is simply
    Gaussian noise. The current method employed is the Davies-Harte method, which fails for H ≈ 0.

    Looking for help to implement a Cholesky decomposition method and the Hosking's method.
    This is based on https://github.com/LRydin/MFDFA/blob/master/MFDFA/fgn.py and the work of
    Christopher Flynn fbm in https://github.com/crflynn/fbm

    See also Davies, Robert B., and D. S. Harte. 'Tests for Hurst effect.' Biometrika 74, no.1
    (1987): 95-101.

    Parameters
    ----------
    size : int
        Length of fractional Gaussian noise to generate.
    hurst_exponent : float
        Hurst exponent H in (0,1).

    Returns
    -------
    array
        Simulated complexity time series.

    """
    # Sanity checks
    assert isinstance(size, int), "Size must be an integer number"
    assert isinstance(hurst_exponent, float), "Hurst index must be a float in (0,1)"

    # Generate linspace
    k = np.linspace(0, size - 1, size)

    # Correlation function
    cor = 0.5 * (
        np.abs(k - 1) ** (2 * hurst_exponent)
        - 2 * np.abs(k) ** (2 * hurst_exponent)
        + np.abs(k + 1) ** (2 * hurst_exponent)
    )

    # Eigenvalues of the correlation function
    eigenvals = np.sqrt(np.fft.fft(np.concatenate([cor[:], 0, cor[1:][::-1]], axis=None).real))

    # Two normal distributed noises to be convoluted
    gn = np.random.normal(0.0, 1.0, size)
    gn2 = np.random.normal(0.0, 1.0, size)

    # This is the Davies–Harte method
    w = np.concatenate(
        [
            (eigenvals[0] / np.sqrt(2 * size)) * gn[0],
            (eigenvals[1:size] / np.sqrt(4 * size)) * (gn[1:] + 1j * gn2[1:]),
            (eigenvals[size] / np.sqrt(2 * size)) * gn2[0],
            (eigenvals[size + 1 :] / np.sqrt(4 * size)) * (gn[1:][::-1] - 1j * gn2[1:][::-1]),
        ],
        axis=None,
    )

    # Perform fft. Only first N entry are useful
    f = np.fft.fft(w).real[:size] * ((1.0 / size) ** hurst_exponent)

    return f


def _complexity_simulate_randomwalk(size=1000):
    """Random walk."""
    steps = np.random.choice(a=[-1, 0, 1], size=size - 1)
    return np.concatenate([np.zeros(1), steps]).cumsum(0)

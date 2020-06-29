# -*- coding: utf-8 -*-
import numpy as np


def complexity_simulate(duration=10, sampling_rate=1000, method="ornstein", hurst_exponent=0.5, **kwargs):
    """Simulate chaotic time series.

    Generates time series using the discrete approximation of the
    Mackey-Glass delay differential equation described by Grassberger &
    Procaccia (1983).

    Parameters
    ----------
    duration : int
        Desired length of duration (s).
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    duration : int
        The desired length in samples.
    method : str
        The method. can be 'hurst' for a (fractional) Ornstein–Uhlenbeck process or 'mackeyglass' to
        use the Mackey-Glass equation.
    hurst_exponent : float
        Defaults to 0.5.
    **kwargs
        Other arguments.

    Returns
    -------
    array
        Simulated complexity time series.

    Examples
    ------------
    >>> import neurokit2 as nk
    >>>
    >>> signal1 = nk.complexity_simulate(duration=30, sampling_rate=100, method="ornstein")
    >>> signal2 = nk.complexity_simulate(duration=30, sampling_rate=100, method="mackeyglass")
    >>> nk.signal_plot([signal1, signal2])

    Returns
    -------
    x : array
        Array containing the time series.

    """
    method = method.lower()
    if method in ["fractal", "fractional", "husrt", "ornsteinuhlenbeck", "ornstein"]:
        signal = _complexity_simulate_ornstein(
            duration=duration, sampling_rate=sampling_rate, hurst_exponent=hurst_exponent, **kwargs
        )
    else:
        signal = _complexity_simulate_mackeyglass(duration=duration, sampling_rate=sampling_rate, **kwargs)
    return signal


# =============================================================================
# Methods
# =============================================================================
def _complexity_simulate_mackeyglass(
    duration=10, sampling_rate=1000, x0=None, a=0.2, b=0.1, c=10.0, n=1000, discard=250
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
        Initial condition for the discrete map. Should be of length n. Defaults to None.
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
    grids = n * discard + sampling_rate * length
    x = np.empty(grids)

    if not x0:
        x[:n] = 0.5 + 0.05 * (-1 + 2 * np.random.random(n))
    else:
        x[:n] = x0

    A = (2 * n - b * tau) / (2 * n + b * tau)
    B = a * tau / (2 * n + b * tau)

    for i in range(n - 1, grids - 1):
        x[i + 1] = A * x[i] + B * (x[i - n] / (1 + x[i - n] ** c) + x[i - n + 1] / (1 + x[i - n + 1] ** c))
    return x[n * discard :: sampling_rate]


def _complexity_simulate_ornstein(duration=10, sampling_rate=1000, theta=0.3, sigma=0.1, hurst_exponent=0.7):
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
    dB = (duration ** hurst_exponent) * _complexity_simulate_fractionalnoise(size=length, hurst_exponent=hurst_exponent)

    # Initialise the array y
    y = np.zeros([length])

    # Integrate the process
    for i in range(1, length):
        y[i] = y[i - 1] - theta * y[i - 1] * (1 / sampling_rate) + sigma * dB[i]
    return y


def _complexity_simulate_fractionalnoise(size=1000, hurst_exponent=0.5):
    """Generates fractional Gaussian noise.

    This is based on https://github.com/LRydin/MFDFA/blob/master/MFDFA/fgn.py and the work of Christopher Flynn fbm in
    https://github.com/crflynn/fbm and Davies, Robert B., and D. S. Harte. 'Tests for Hurst effect.' Biometrika 74, no.1
    (1987): 95-101.

    Generates fractional Gaussian noise with a Hurst index H in (0,1). If H = 1/2 this is simply Gaussian
    noise. The current method employed is the Davies–Harte method, which fails for H ≈ 0. A Cholesky
    decomposition method and the Hosking’s method will be implemented in later versions.

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

import numpy as np
import pandas as pd
import scipy.integrate

from ..stats import density


def entropy_power(signal, **kwargs):
    """**Entropy Power (PowEn)**

    The Shannon Entropy Power (PowEn or SEP) is a measure of the effective variance of a random
    variable: it is the variance of the Gaussian variable that has the same (differential) entropy
    (Shannon, 1948; Dembo et al., 1991). It is defined as:

    .. math:: N_X = \\frac{1}{2 \\pi e} e^{2 H_X}

    where :math:`H_X` is the :func:`differential entropy <entropy_differential>` (in nats) of the
    distribution of the signal. It is expressed in the squared units of the signal, and it is equal
    to the variance for a Gaussian distribution and lower for any other distribution (e.g.,
    :math:`6 / (\\pi e) \\approx 0.70` times the variance for a uniform distribution).

    It is based on the estimation of the density of the variable, thus relying on :func:`density`:
    :math:`H_X = -\\int f(x) \\ln f(x) dx` is integrated (Simpson's rule) from a Gaussian kernel
    density estimate evaluated between the minimum and the maximum of the signal, and normalized
    to integrate to 1 over this range (so that multiplying the signal by *a* multiplies PowEn by
    :math:`a^2`). The estimate therefore depends on the bandwidth of the kernel.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    **kwargs
        Other arguments to be passed to :func:`density`, such as ``bandwidth`` (see
        :func:`density_bandwidth`) or ``desired_length`` (the number of points at which the density
        is evaluated, 100 by default).

    Returns
    -------
    powen : float
        The computed entropy power measure, in the squared units of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used, such as the
        estimated (normalized) ``Density`` and the ``Values`` at which it was evaluated.

    See Also
    --------
    fishershannon_information, entropy_differential, density

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      import matplotlib.pyplot as plt

      signal = nk.signal_simulate(duration=10, frequency=[10, 12], noise=0.1)

      powen, info = nk.entropy_power(signal)
      powen

      # Visualize the distribution that the entropy power is based on
      @savefig entropy_power2.png scale=100%
      plt.plot(info["Values"], info["Density"])
      @suppress
      plt.close()

    Change density bandwidth.

    .. ipython:: python

      powen, info = nk.entropy_power(signal, bandwidth=0.01)
      powen

    For a normally distributed signal, the entropy power is close to the variance.

    .. ipython:: python

      import numpy as np

      signal = np.random.default_rng(42).normal(0, 2, size=1000)
      powen, info = nk.entropy_power(signal)
      powen, np.var(signal)

    References
    ----------
    * Guignard, F., Laib, M., Amato, F., & Kanevski, M. (2020). Advanced analysis of temporal data
      using Fisher-Shannon information: theoretical development and application in geosciences.
      Frontiers in Earth Science, 8, 255.
    * Vignat, C., & Bercher, J. F. (2003). Analysis of signals in the Fisher-Shannon information
      plane. Physics Letters A, 312(1-2), 27-33.
    * Dembo, A., Cover, T. M., & Thomas, J. A. (1991). Information theoretic inequalities. IEEE
      Transactions on Information theory, 37(6), 1501-1518.
    * Shannon, C. E. (1948). A mathematical theory of communication. The Bell system technical
      journal, 27(3), 379-423.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    signal = np.asarray(signal, dtype=float)
    if np.any(~np.isfinite(signal)):
        raise ValueError("NeuroKit error: entropy_power(): the signal contains missing or infinite values.")
    if len(np.unique(signal)) < 2:
        raise ValueError(
            "NeuroKit error: entropy_power(): the signal should have at least 2 different values to"
            " estimate its density (the entropy power of a constant signal is 0)."
        )

    # we consider a random variable x whose probability density function is denoted as fx
    x_range, fx = density(signal, **kwargs)

    valid = np.where(fx > 0)[0]
    x_range, fx = x_range[valid], fx[valid]

    # The density is only evaluated between the minimum and the maximum of the signal: normalize it
    # so that it integrates to 1 over this range (otherwise, the missing mass of the tails makes the
    # entropy depend on the units of the signal)
    fx = fx / scipy.integrate.simpson(fx, x=x_range)

    # Differential entropy (in nats) of the estimated density
    H = -1 * scipy.integrate.simpson(fx * np.log(fx), x=x_range)

    # Entropy power: the variance of the Gaussian with the same entropy, exp(2H) / (2 pi e). In
    # Dembo et al. (1991), 2H is divided by n, the dimension of the random vector (1 here), not by
    # the number of samples.
    powen = np.exp(2 * H) / (2 * np.pi * np.e)

    return powen, {"Density": fx, "Values": x_range}

import numpy as np
import pandas as pd
import scipy.integrate

from ..stats import density


def entropy_power(signal, **kwargs):
    """**Entropy Power (PowEn)**

    The Shannon Entropy Power (PowEn or SEP) is a measure of the effective variance of a random
    vector. It is based on the estimation of the density of the variable, thus relying on :func:`density`.

    .. warning::

        We are not sure at all about the correct implementation of this function. Please consider
        helping us by double-checking the code against the formulas in the references.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    **kwargs
        Other arguments to be passed to :func:`density_bandwidth`.

    Returns
    -------
    powen : float
        The computed entropy power measure.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    information_fisershannon

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

    References
    ----------
    * Guignard, F., Laib, M., Amato, F., & Kanevski, M. (2020). Advanced analysis of temporal data
      using Fisher-Shannon information: theoretical development and application in geosciences.
      Frontiers in Earth Science, 8, 255.
    * Vignat, C., & Bercher, J. F. (2003). Analysis of signals in the Fisher-Shannon information
      plane. Physics Letters A, 312(1-2), 27-33.
    * Dembo, A., Cover, T. M., & Thomas, J. A. (1991). Information theoretic inequalities. IEEE
      Transactions on Information theory, 37(6), 1501-1518.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # we consider a random variable x whose probability density function is denoted as fx
    x_range, fx = density(signal, **kwargs)

    valid = np.where(fx > 0)[0]

    # In https://github.com/fishinfo/FiShPy/blob/master/FiSh.py
    # The formula is somewhat different...
    # And on top of that it looks like it also differs between Dembo 1991 and Vignat 2003
    # (The former divides by n)

    # Shannon Entropy (https://en.wikipedia.org/wiki/Entropy_power_inequality)
    H = fx[valid] * np.log(fx[valid])
    H = -1 * scipy.integrate.simpson(H, x=x_range[valid])

    # Entropy power
    powen = np.exp(2 * H / len(signal)) / (2 * np.pi * np.e)

    return powen, {"Density": fx[valid], "Values": x_range[valid]}

import numpy as np
import scipy.integrate

from .entropy_power import entropy_power


def fishershannon_information(signal, **kwargs):
    """**Fisher-Shannon Information (FSI)**

    The :func:`Shannon Entropy Power <entropy_power>` is closely related to another index, the
    Fisher Information Measure (FIM). Their combination results in the Fisher-Shannon Information
    index, or Fisher-Shannon Complexity (Vignat & Bercher, 2003):

    .. math::

        FSC = N_X \\cdot J_X, \\quad J_X = \\int \\frac{f'(x)^2}{f(x)} dx

    where :math:`N_X` is the entropy power and :math:`J_X` the Fisher information of the density
    *f* of the signal (estimated by :func:`density`). FSC does not depend on the units of the
    signal and is at least 1, the value of a Gaussian distribution (Stam's inequality). The kernel
    density estimate smooths the density, so that values for non-Gaussian distributions are
    underestimated (and can be below 1 for distributions with sharp edges, such as uniform).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    **kwargs
        Other arguments to be passed to :func:`density`, such as ``bandwidth`` (see
        :func:`density_bandwidth`).

    Returns
    -------
    fsi : float
        The computed FSI.
    info : dict
        A dictionary containing additional information regarding the parameters used, such as the
        Fisher information ``FI``.

    See Also
    --------
    entropy_power

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=10, frequency=[10, 12], noise=0.1)

      fsi, info = nk.fishershannon_information(signal)
      fsi


    References
    ----------
    * Guignard, F., Laib, M., Amato, F., & Kanevski, M. (2020). Advanced analysis of temporal data
      using Fisher-Shannon information: theoretical development and application in geosciences.
      Frontiers in Earth Science, 8, 255.
    * Vignat, C., & Bercher, J. F. (2003). Analysis of signals in the Fisher-Shannon information
      plane. Physics Letters A, 312(1-2), 27-33.

    """
    # Shannon Power Entropy
    powen, info = entropy_power(signal, **kwargs)
    x_range = info["Values"]
    fx = info["Density"]
    gx = np.gradient(fx, x_range)  # The derivative with respect to x, not to the index

    # Fisher
    fi = gx**2 / fx
    fi = scipy.integrate.simpson(fi, x=x_range)
    info.update({"FI": fi})

    # Fisher-Shannon Complexity
    fsc = powen * fi

    # if fsc < 1:
    #     warnings.warn(
    #         "Fisher-Shannon Complexity is lower than 1. The problem could be related to kernel"
    #         " density estimation, bandwidth selection, or too little data points."
    #     )

    return fsc, info

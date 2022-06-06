import numpy as np
import scipy.integrate

from .entropy_power import entropy_power


def fishershannon_information(signal, **kwargs):
    """**Fisher-Shannon Information (FSI)**

    The :func:`Shannon Entropy Power <entropy_power>` is closely related to another index, the
    Fisher Information Measure (FIM). Their combination results in the Fisher-Shannon Information
    index.

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
    fsi : float
        The computed FSI.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_power

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=10, frequency=[10, 12], noise=0.1)

      fsi, info = nk.fishershannon_information(signal, method=0.01)
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
    gx = np.gradient(fx)

    # Fisher
    fi = gx ** 2 / fx
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

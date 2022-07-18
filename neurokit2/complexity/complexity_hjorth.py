import numpy as np
import pandas as pd


def complexity_hjorth(signal):
    """**Hjorth's Complexity and Parameters**

    Hjorth Parameters are indicators of statistical properties initially introduced by Hjorth
    (1970) to describe the general characteristics of an EEG trace in a few quantitative terms, but
    which can applied to any time series. The parameters are activity, mobility, and complexity.
    NeuroKit returns complexity directly in the output tuple, but the other parameters can be found
    in the dictionary.

    * The **activity** parameter is simply the variance of the signal, which corresponds to the
      mean power of a signal (if its mean is 0).

      .. math::

        Activity = \\sigma_{signal}^2

    * The **mobility** parameter represents the mean frequency or the proportion of standard
      deviation of the power spectrum. This is defined as the square root of variance of the
      first derivative of the signal divided by the variance of the signal.

      .. math::

        Mobility = \\frac{\\sigma_{dd}/ \\sigma_{d}}{Complexity}

    * The **complexity** parameter gives an estimate of the bandwidth of the signal, which
      indicates the similarity of the shape of the signal to a pure sine wave (for which the
      value converges to 1). In other words, it is a measure of the "excessive details" with
      reference to the "softest" possible curve shape. The Complexity parameter is defined as the
      ratio of the mobility of the first derivative of the signal to the mobility of the signal.

      .. math::

        Complexity = \\sigma_{d}/ \\sigma_{signal}

    :math:`d` and :math:`dd` represent the first and second derivatives of the signal, respectively.

    Hjorth (1970) illustrated the parameters as follows:

    .. figure:: ../img/hjorth1970.png
       :alt: Figure from Hjorth (1970).
       :target: http://dx.doi.org/10.1016/0013-4694(70)90143-4


    See Also
    --------
    .fractal_petrosian

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.


    Returns
    -------
    hjorth : float
        Hjorth's Complexity.
    info : dict
        A dictionary containing the additional Hjorth parameters, such as ``"Mobility"`` and
        ``"Activity"``.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a signal with duration os 2s
      signal = nk.signal_simulate(duration=2, frequency=5)

      # Compute Hjorth's Complexity
      complexity, info = nk.complexity_hjorth(signal)
      complexity
      info

    References
    ----------
    * Hjorth, B (1970) EEG Analysis Based on Time Domain Properties. Electroencephalography and
      Clinical Neurophysiology, 29, 306-310.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Calculate derivatives
    dx = np.diff(signal)
    ddx = np.diff(dx)

    # Calculate variance and its derivatives
    x_var = np.var(signal)  # = activity
    dx_var = np.var(dx)
    ddx_var = np.var(ddx)

    # Mobility and complexity
    mobility = np.sqrt(dx_var / x_var)
    complexity = np.sqrt(ddx_var / dx_var) / mobility
    return complexity, {"Mobility": mobility, "Activity": x_var}

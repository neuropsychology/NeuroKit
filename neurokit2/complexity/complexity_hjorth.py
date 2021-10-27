import numpy as np
import pandas as pd


def complexity_hjorth(signal):
    """Hjorth's Complexity and Parameters

    Hjorth Parameters are indicators of statistical properties used in signal processing in the
    time domain introduced by Hjorth (1970). The parameters are activity, mobility, and complexity.
    NeuroKit returns complexity directly in the output tuple, but the other parameters can be found
    in the dictionary.

    - The **complexity** parameter gives an estimate of the bandwidth of the signal, which
    indicates the similarity of the shape of the signal to a pure sine wave (where the value
    converges to 1). Complexity is define as the ratio of the mobility of the first derivative of
    the signal to the mobility of the signal.
    - The **mobility** parameter represents the mean frequency or the proportion of standard
    deviation of the power spectrum. This is defined as the square root of variance of the first
    derivative of the signal divided by the variance of the signal.
    - The **activity** parameter is simply the variance of the signal.

    See Also
    --------
    fractal_petrosian

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    Returns
    -------
    hjorth : float
         Hjorth's Complexity.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute Hjorth's Complexity.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> complexity, info = nk.complexity_hjorth(signal)
    >>> complexity #doctest: +SKIP

    References
    ----------
    - https://github.com/raphaelvallat/antropy/blob/master/antropy

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

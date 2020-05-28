# -*- coding: utf-8 -*-
import pandas as pd


def eda_autocor(eda_signal, sampling_rate=1000, lag=4):
    """Computes autocorrelation measure of raw EDA signal i.e., the
    correlation between the time series data and a specified time-lagged
    version of itself.

    Parameters
    ----------
    eda_signal : list, array or Series
        The raw EDA signal.
    sampling_rate : int
        The sampling frequency of raw EDA signal (in Hz, i.e., samples/second).
    lag : int
        Time lag in seconds. Defaults to 4 seconds to avoid autoregressive correlations
        approaching 1, as recommended by Halem et al. (2020)

    Returns
    -------
    float
        Autocorrelation index of the eda signal.

    See Also
    --------
    eda_simulate


    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> # Simulate EDA signal
    >>> eda_signal = nk.eda_simulate(duration=5, scr_number=5, drift=0.1)
    >>> cor = nk.eda_autocor(eda_signal)
    >>> cor #doctest: +SKIP

    References
    -----------
    - Halem, S., van Roekel, E., Kroencke, L., Kuper, N., & Denissen, J. (2020). Moments That Matter? On the Complexity of Using Triggers Based on Skin Conductance to Sample Arousing Events Within an Experience Sampling Framework. European Journal of Personality.
    """
    # Sanity checks
    if not isinstance(eda_signal, pd.Series):
        eda_signal = pd.Series(eda_signal)

    lag_samples = lag*sampling_rate

    if lag_samples > len(eda_signal):
        raise ValueError("NeuroKit error: eda_autocor(): The time lag "
                         "exceeds the duration of the EDA signal. "
                         "Consider using a longer duration of the EDA signal.")

    cor = eda_signal.autocorr(lag=lag_samples)

    return cor

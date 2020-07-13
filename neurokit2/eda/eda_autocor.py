# -*- coding: utf-8 -*-
import pandas as pd

from ..signal import signal_autocor


def eda_autocor(eda_cleaned, sampling_rate=1000, lag=4):
    """Computes autocorrelation measure of raw EDA signal i.e., the correlation between the time series data and a
    specified time-lagged version of itself.

    Parameters
    ----------
    eda_cleaned : Union[list, np.array, pd.Series]
        The cleaned EDA signal.
    sampling_rate : int
        The sampling frequency of raw EDA signal (in Hz, i.e., samples/second). Defaults to 1000Hz.
    lag : int
        Time lag in seconds. Defaults to 4 seconds to avoid autoregressive
        correlations approaching 1, as recommended by Halem et al. (2020).

    Returns
    -------
    float
        Autocorrelation index of the eda signal.

    See Also
    --------
    eda_simulate, eda_clean


    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> # Simulate EDA signal
    >>> eda_signal = nk.eda_simulate(duration=5, scr_number=5, drift=0.1)
    >>> eda_cleaned = nk.eda_clean(eda_signal)
    >>> cor = nk.eda_autocor(eda_cleaned)
    >>> cor #doctest: +SKIP

    References
    -----------
    - Halem, S., van Roekel, E., Kroencke, L., Kuper, N., & Denissen, J. (2020). Moments That Matter?
      On the Complexity of Using Triggers Based on Skin Conductance to Sample Arousing Events Within
      an Experience Sampling Framework. European Journal of Personality.

    """
    # Sanity checks
    if isinstance(eda_cleaned, pd.DataFrame):
        colnames = eda_cleaned.columns.values
        if len([i for i in colnames if "EDA_Clean" in i]) == 0:
            raise ValueError("NeuroKit error: eda_autocor(): Your input does not contain the cleaned EDA signal.")
        else:
            eda_cleaned = eda_cleaned["EDA_Clean"]
    if isinstance(eda_cleaned, pd.Series):
        eda_cleaned = eda_cleaned.values

    # Autocorrelation
    lag_samples = lag * sampling_rate

    if lag_samples > len(eda_cleaned):
        raise ValueError(
            "NeuroKit error: eda_autocor(): The time lag "
            "exceeds the duration of the EDA signal. "
            "Consider using a longer duration of the EDA signal."
        )

    cor = signal_autocor(eda_cleaned, lag=lag_samples)

    return cor

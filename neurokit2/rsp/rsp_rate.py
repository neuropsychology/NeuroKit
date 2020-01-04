# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from ..signal import signal_interpolate
from ..signal import signal_smooth


def rsp_rate(peaks, sampling_rate=1000, desired_length=None,
             method="khodadad2018"):
    """Compute respiration (RSP) rate.

    Compute respiration rate with the specified method.

    Parameters
    ----------
    peaks : list, array, DataFrame, Series or dict
        The samples at which the inhalation peaks occur. If a dict or a
        DataFrame is passed, it is assumed that these containers were obtained
        with `rsp_findpeaks()`.
    sampling_rate : int
        The sampling frequency of the signal that contains the peaks (in Hz,
        i.e., samples/second).
    desired_length : int
        By default, the returned respiration rate has the same number of
        elements as `peaks`. If set to an integer, the returned rate will be
        interpolated between `peaks` over `desired_length` samples. Has no
        effect if a DataFrame is passed in as the `peaks` argument.
    method : str
        The processing pipeline to apply. Can be one of "khodadad2018"
        (default) or "biosppy".

    Returns
    -------
    signals : DataFrame
        A DataFrame containing the respiration rate accessible with the key
        "RSP_Rate".

    See Also
    --------
    rsp_clean, rsp_findpeaks, rsp_amplitude, rsp_process, rsp_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration=90, respiratory_rate=15)
    >>> cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    >>> signals, info = nk.rsp_findpeaks(cleaned)
    >>>
    >>> data = nk.rsp_rate(signals)
    >>> data["RSP_Signal"] = cleaned  # Add the signal back
    >>> data.plot(subplots=True)
    """
    # Sanity checks.
    if isinstance(peaks, dict):
        peaks = peaks["RSP_Peaks"]
    elif isinstance(peaks, pd.DataFrame):
        desired_length = len(peaks["RSP_Peaks"])
        peaks = np.where(peaks["RSP_Peaks"] == 1)[0]
    if len(peaks) <= 3:
        print("NeuroKit warning: rsp_rate(): too few peaks detected to "
              "compute the rate.")
        return

    # Find length of final signal to return.
    if desired_length is None:
        desired_length = len(peaks)

    # Calculate period in msec, based on horizontal peak to peak difference and
    # make sure that rate has the same number of elements as peaks (important
    # for interpolation later) by prepending the mean of all periods.
    period = np.ediff1d(peaks, to_begin=0) / sampling_rate
    period[0] = np.mean(period)
    rate = 60 / period

    # Preprocessing.
    rate, peaks = _rsp_rate_preprocessing(rate, peaks, method=method)

    # Interpolate rates to desired_length samples.
    rate = signal_interpolate(rate, x_axis=peaks,
                              desired_length=desired_length)

    signals = pd.DataFrame(rate, columns=["RSP_Rate"])

    return signals


# =============================================================================
# Internals
# =============================================================================
def _rsp_rate_preprocessing(rate, peaks, troughs=None, method="biosppy"):

    method = method.lower()  # remove capitalised letters
    if method == "biosppy":
        rate, peaks = _rsp_rate_outliers(rate, peaks, threshold_absolute=35)
        # Smooth with moving average
        rate = signal_smooth(signal=rate, kernel='boxcar', size=3)
    elif method == "khodadad2018":
        pass
    else:
        raise ValueError("NeuroKit error: rsp_rate(): 'method' should be "
                         "one of 'khodadad2018' or 'biosppy'.")

    return rate, peaks


def _rsp_rate_outliers(rate, peaks, troughs=None, threshold_absolute=35):

    if threshold_absolute is None:
        return rate, peaks

    # Enforce physiological limits.
    keep = np.nonzero(rate <= threshold_absolute)

    return rate[keep], peaks[keep]

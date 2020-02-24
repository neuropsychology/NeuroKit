# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal.signal_rate import signal_rate
from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize
from ..signal import signal_interpolate
from ..signal import signal_smooth




def rsp_rate(peaks, sampling_rate=1000, desired_length=None, method="khodadad2018"):
    """Compute respiration (RSP) rate.

    Compute respiration rate with the specified method.

    Parameters
    ----------
    peaks : list, array, DataFrame, Series or dict
        The samples at which the inhalation peaks occur. If a dict or a
        DataFrame is passed, it is assumed that these containers were obtained
        with `rsp_peaks()`.
    sampling_rate : int
        The sampling frequency of the signal that contains the peaks (in Hz,
        i.e., samples/second). Defaults to 1000Hz.
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
    array
        A vector containing the respiration rate.

    See Also
    --------
    rsp_clean, rsp_peaks, rsp_amplitude, rsp_process, rsp_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration=90, respiratory_rate=15)
    >>> cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    >>> info, peak_signal = nk.rsp_peaks(cleaned)
    >>>
    >>> rate = nk.rsp_rate(peak_signal, desired_length=len(rsp))
    >>> nk.signal_plot([rsp, rate], subplots=True)
    """
    # Format input.
    peaks, desired_length = _signal_formatpeaks_sanitize(peaks, desired_length)

    # Get rate values
    rate = signal_rate(peaks, sampling_rate, desired_length=len(peaks))

    # Preprocessing.
    rate, peaks = _rsp_rate_preprocessing(rate, peaks, method=method)

    # Interpolate rates to desired_length samples.
    rate = signal_interpolate(peaks, rate, desired_length=desired_length, method='quadratic')

    return rate


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

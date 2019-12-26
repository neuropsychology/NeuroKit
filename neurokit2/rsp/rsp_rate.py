# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_interpolate


def rsp_rate(peaks, troughs=None, sampling_rate=1000, desired_length=None):
    """Calculate respiration (RSP) rate.

    Calculate respiration rate and amplitude.

    Parameters
    ----------
    peaks : list, array, DataFrame, Series or dict
        The samples at which the inhalation peaks occur. If a dict or a
        DataFrame is passed, it is assumed that these containers were obtained
        with `rsp_findpeaks()`.
    troughs : list, array, or Series
        The samples at which the exhalation troughs occur. Can be passed in
        individually, or is automatically inferred if peaks is a dict or
        DataFrame obtained with `rsp_findpeaks()`.
    sampling_rate : int
        The sampling frequency of the signal that contains the peaks and
        troughs (in Hz, i.e., samples/second).
    desired_length : int
        By default, the returned respiration rate, period, and amplitude each
        have the same number of elements as peaks. If set to an integer, each
        of the returned elements will be interpolated between peaks over
        desired_length samples. Has not effect if a DataFrame is passed in as
        the peaks argument.

    Returns
    -------
    signals : DataFrame
        A DataFrame containing respiration rate, and amplitude,
        accessible with the keys 'RSP_Rate' and 'RSP_Amplitude'
        respectively.

    See Also
    --------
    rsp_clean, rsp_findpeaks, rsp_process, rsp_plot

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> rsp = np.cos(np.linspace(start=0, stop=50, num=10000))
    >>> signals, info = nk.rsp_findpeaks(rsp)
    >>>
    >>> data = nk.rsp_rate(signals)
    >>> data["RSP_Signal"] = rsp  # Add the signal back
    >>> nk.standardize(data).plot()
    """
    if isinstance(peaks, dict):
        troughs = peaks["RSP_Troughs"]
        peaks = peaks["RSP_Peaks"]
    elif isinstance(peaks, pd.DataFrame):
        desired_length = len(peaks["RSP_Peaks"])
        troughs = np.where(peaks["RSP_Troughs"] == 1)[0]
        peaks = np.where(peaks["RSP_Peaks"] == 1)[0]

    # Calculate period in msec, based on horizontal peak to peak
    # difference. Make sure that period has the same number of elements as
    # peaks (important for interpolation later) by prepending the mean of
    # all periods.
    period = np.ediff1d(peaks, to_begin=0) / sampling_rate
    period[0] = np.mean(period)
    rate = 60 / period

    # Interpolate all statistics to length of the breathing signal.
    if desired_length is None:
        desired_length = len(peaks)


    rate = signal_interpolate(rate, x_axis=peaks,
                              desired_length=desired_length)

    # Prepare output.
    out = {"RSP_Rate": rate}

    # Add amplitude if troughs are available.
    if troughs is not None:
        # TODO: normalize amplitude?
        amplitude = peaks - troughs
        out["RSP_Amplitude"] = signal_interpolate(amplitude, x_axis=peaks,
                                                  desired_length=desired_length)

    signals = pd.DataFrame.from_dict(out)
    return(signals)

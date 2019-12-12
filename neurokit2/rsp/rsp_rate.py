# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_interpolate


def rsp_rate(peaks, troughs=None, return_amplitude=True, sampling_rate=1000,
             desired_length=None):
    """Calculate respiration (RSP) rate.

    Calculate respiration rate, as well as the -period and -amplitude.

    Parameters
    ----------
    peaks : list, array, Series or dict
        The samples at which the inhalation peaks occur. If a dict is passed
        it is assumed that the dict is obtained with `rsp_findpeaks`.
    troughs : list, array, or Series, default None
        The samples at which the exhalation troughs occur. Only relevant if
        return_amplitude is True.
    return_amplitude : bool, default True
        Calculate and return breathing amplitude if True.
    sampling_rate : int, default 1000
        The sampling frequency of the signal that contains the peaks and
        troughs (in Hz, i.e., samples/second).
    desired_length : int, default None
        By default, the returned respiration rate, period, and amplitude each
        have the same number of elements as peaks. If set to an integer, each
        of the returned elements will be interpolated between peaks over
        desired_length samples.

    Returns
    -------
    DataFrame
        A DataFrame containing respiration rate, period, and amplitude,
        accessible with the keys "RSP_Rate", "RSP_Period" and "RSP_Amplitude"
        respectively.

    See Also
    --------
    rsp_findpeaks, rsp_rate, rsp_process, rsp_plot

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=50, num=10000))
    >>> peaks_data, peaks_info = nk.rsp_findpeaks(signal)
    >>>
    >>> data = nk.rsp_rate(peaks_info, desired_length=len(signal))
    >>> data["RSP_Signal"] = signal  # Add the signal back
    >>> nk.standardize(data).plot()
    """
    if isinstance(peaks, dict):
        troughs = peaks["RSP_Troughs"]
        peaks = peaks["RSP_Peaks"]

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

    period = signal_interpolate(period, x_axis=peaks,
                                desired_length=desired_length)
    rate = signal_interpolate(rate, x_axis=peaks,
                              desired_length=desired_length)

    # Prepare output.
    out = {"RSP_Rate": rate,
           "RSP_Period": period}

    # Add amplitude if troughs are available.
    if (troughs is not None) and return_amplitude:
        # TODO: normalize amplitude?
        amplitude = peaks - troughs
        out["RSP_Amplitude"] = signal_interpolate(amplitude, x_axis=peaks,
                                                  desired_length=desired_length)

    out = pd.DataFrame.from_dict(out)
    return(out)

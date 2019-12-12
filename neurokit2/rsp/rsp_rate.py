# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd




def rsp_rate(peaks, troughs=None, sampling_rate=1000, desired_length=None):
    """Get Respiration rate.

    Calculate RSP (respiration) rate based on horizontal peak to peak difference, as well as the period and the amplitude.

    Examples
    ---------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=50, num=10000))
    >>> peaks_info = nk.rsp_findpeaks(signal)
    >>>
    >>> results = nk.rsp_rate(peaks_info, desired_length=len(signal))
    >>> results["RSP_Signal"] = signal
    >>> nk.standardize(results).plot()
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

    period = signal_interpolate(period, x_axis=peaks, desired_length=desired_length)
    rate = signal_interpolate(rate, x_axis=peaks, desired_length=desired_length)

     # Prepare output
    out = {"RSP_Rate": rate,
           "RSP_Period": period}

    # Add amplitude if troughs are available
    if troughs is not None:
        # TODO: normalize amplitude?
        amplitude = peaks - troughs
        out["RSP_Amplitude"] = signal_interpolate(amplitude, x_axis=peaks, desired_length=desired_length)

    out = pd.DataFrame.from_dict(out)
    return(out)



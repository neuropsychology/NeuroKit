# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.signal

from ..signal import signal_filter
from ..signal import signal_interpolate









def rsp_preprocessing(rsp, sampling_rate=1000, outlier_threshold=1/3):
    """

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> rsp = np.cos(np.linspace(start=0, stop=40, num=20000))
    >>> preprocessed = nk.rsp_preprocessing(rsp, sampling_rate=1000, outlier_threshold=1/5)
    >>> nk.plot_events_in_signal(preprocessed['RSP_data'], preprocessed['RSP_info']["RSP_Peaks"])
    """
    # Detrend and lowpass-filter the signal to be able to reliably detect
    # zero crossings in raw signal.
    rsp = scipy.signal.detrend(rsp, type="linear")
    filtered = signal_filter(rsp, sampling_rate=sampling_rate, highcut=2, method="butterworth")

    # Detect zero crossings (note that these are zero crossings in the raw
    # signal, not in its gradient).
    greater = filtered > 0
    smaller = filtered < 0
    rise_x = np.where(np.bitwise_and(smaller[:-1], greater[1:]))[0]
    fall_x = np.where(np.bitwise_and(greater[:-1], smaller[1:]))[0]

    if rise_x[0] < fall_x[0]:
        start_x = "rise"
    elif fall_x[0] < rise_x[0]:
        start_x = "fall"

    x_axis = np.concatenate((rise_x, fall_x))
    x_axis.sort(kind="mergesort")

    # Find extrema by searching minima between falling zero crossing and
    # rising zero crossing, and searching maxima between rising zero
    # crossing and falling zero crossing.
    extrema = []
    for i in range(len(x_axis) - 1):

        # Determine whether to search for minimum or maximum.
        if start_x == "rise":
            if (i + 1) % 2 != 0:
                argextreme = np.argmax
            else:
                argextreme = np.argmin
        elif start_x == "fall":
            if (i + 1) % 2 != 0:
                argextreme = np.argmin
            else:
                argextreme = np.argmax

        # Get the two zero crossings between which the extreme will be
        # searched.
        beg = x_axis[i]
        end = x_axis[i + 1]

        extreme = argextreme(filtered[beg:end])
        extrema.append(beg + extreme)

    extrema = np.asarray(extrema)

    # Only consider those extrema that have a minimum vertical distance
    # to their direct neighbor, i.e., define outliers in absolute amplitude
    # difference between neighboring extrema.
    vertdiff = np.abs(np.diff(filtered[extrema]))
    avgvertdiff = np.mean(vertdiff)
    minvert = np.where(vertdiff > avgvertdiff * outlier_threshold)[0]
    extrema = extrema[minvert]

    # Make sure that the alternation of peaks and troughs is unbroken. If
    # alternation of sign in extdiffs is broken, remove the extrema that
    # cause the breaks.
    amps = filtered[extrema]
    extdiffs = np.sign(np.diff(amps))
    extdiffs = np.add(extdiffs[0:-1], extdiffs[1:])
    removeext = np.where(extdiffs != 0)[0] + 1
    extrema = np.delete(extrema, removeext)
    amps = np.delete(amps, removeext)

    # To be able to consistently calculate breathing amplitude, make
    # sure that the extrema always start with a trough and end with a peak,
    # since breathing amplitude will be defined as vertical distance
    # between each peak and the preceding trough. Note that this also
    # ensures that the number of peaks and troughs is equal.
    if amps[0] > amps[1]:
        extrema = np.delete(extrema, 0)
    if amps[-1] < amps[-2]:
        extrema = np.delete(extrema, -1)
    peaks = extrema[1::2]
    troughs = extrema[0:-1:2]

    # Prepare output
    data = pd.DataFrame({"RSP_Raw": rsp,
                         "RSP_Filtered": filtered})
    info = {"RSP_Peaks": peaks,
            "RSP_Troughs": troughs}

    out = {"RSP_data": data,
           "RSP_info": info}
    return(out)
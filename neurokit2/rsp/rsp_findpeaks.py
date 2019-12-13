# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def rsp_findpeaks(rsp_filtered, sampling_rate=1000, outlier_threshold=0.3):
    """Identify extrema in a respiration (RSP) signal.

    Identifies inhalation peaks and exhalation troughs in a zero-centered
    respiration signal. The algorithm is based on (but not an exact
    implementation of) the "Zero-crossing algorithm with amplitude threshold"
    in the following paper:

    https://www.biorxiv.org/content/10.1101/270348v1.full.

    Parameters
    ----------
    rsp_filtered : list, array or Series
        The cleaned respiration channel as returned by `rsp_clean`
    sampling_rate : int, default 1000
        The sampling frequency of rsp_filtered (in Hz, i.e., samples/second).
    outlier_threshold : float, default 0.3
        Extrema that have a vertical distance smaller than (outlier_threshold *
        average vertical distance) to any direct neighbour are removed as
        false positive outliers. I.e., outlier_threshold should be a float with
        positive sign. Larger values of outlier_threshold correspond to more
        conservative thresholds (i.e., more extrema removed as outliers).

    Returns
    -------
    extrema_signal: dict
        Occurences of inhalation peaks and exhalation troughs marked as "1" in
        lists of zeros with the same length as rsp_filtered. Accessible with
        the keys "RSP_Peaks_Signal" and "RSP_Troughs_Signal" respectively.
    extrema: dict
        A dictionary containing the samples at which inhalation peaks and
        exhalation troughs occur, accessible with the keys "RSP_Peaks", and
        "RSP_Troughs" respectively.

    See Also
    --------
    rsp_findpeaks, rsp_rate, rsp_process, rsp_plot

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=40, num=20000))
    >>> data = nk.rsp_clean(signal, sampling_rate=1000)
    >>> peaks_data, peaks = nk.rsp_findpeaks(rsp_filtered=data["RSP_Filtered"], sampling_rate=1000)
    >>> nk.events_plot([peaks["RSP_Peaks"], peaks["RSP_Troughs"]], data)
    """
    # Try retrieving right column
    if isinstance(rsp_filtered, pd.DataFrame):
        try:
            rsp_filtered = rsp_filtered["RSP_Filtered"]
        except NameError:
            try:
                rsp_filtered = rsp_filtered["RSP_Raw"]
            except NameError:
                rsp_filtered = rsp_filtered["RSP"]

    rsp_filtered = np.array(rsp_filtered)

    # Detect zero crossings (note that these are zero crossings in the raw
    # signal, not in its gradient).
    greater = rsp_filtered > 0
    smaller = rsp_filtered < 0
    risex = np.where(np.bitwise_and(smaller[:-1], greater[1:]))[0]
    fallx = np.where(np.bitwise_and(greater[:-1], smaller[1:]))[0]

    if risex[0] < fallx[0]:
        startx = "rise"
    elif fallx[0] < risex[0]:
        startx = "fall"

    allx = np.concatenate((risex, fallx))
    allx.sort(kind="mergesort")

    # Find extrema by searching minima between falling zero crossing and
    # rising zero crossing, and searching maxima between rising zero
    # crossing and falling zero crossing.
    extrema = []
    for i in range(len(allx) - 1):

        # Determine whether to search for minimum or maximum.
        if startx == "rise":
            if (i + 1) % 2 != 0:
                argextreme = np.argmax
            else:
                argextreme = np.argmin
        elif startx == "fall":
            if (i + 1) % 2 != 0:
                argextreme = np.argmin
            else:
                argextreme = np.argmax

        # Get the two zero crossings between which the extreme will be
        # searched.
        beg = allx[i]
        end = allx[i + 1]

        extreme = argextreme(rsp_filtered[beg:end])
        extrema.append(beg + extreme)

    extrema = np.asarray(extrema)

    # Only consider those extrema that have a minimum vertical distance
    # to their direct neighbor, i.e., define outliers in absolute amplitude
    # difference between neighboring extrema.
    vertdiff = np.abs(np.diff(rsp_filtered[extrema]))
    avgvertdiff = np.mean(vertdiff)
    minvert = np.where(vertdiff > (avgvertdiff * outlier_threshold))[0]
    extrema = extrema[minvert]

    # Make sure that the alternation of peaks and troughs is unbroken. If
    # alternation of sign in extdiffs is broken, remove the extrema that
    # cause the breaks.
    amps = rsp_filtered[extrema]
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
    peaks_signal = np.full(len(rsp_filtered), 0)
    peaks_signal[peaks] = 1
    troughs_signal = np.full(len(rsp_filtered), 0)
    troughs_signal[troughs] = 1

    extrema_signal = pd.DataFrame({"RSP_Peaks_Signal": peaks_signal,
                                   "RSP_Troughs_Signal": troughs_signal})

    extrema = {"RSP_Peaks": peaks,
               "RSP_Troughs": troughs}
    return(extrema_signal, extrema)

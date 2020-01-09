# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from ..signal import signal_interpolate


def rsp_amplitude(rsp_signal, extrema, desired_length=None):
    """Compute respiratory amplitude.

    Compute respiratory amplitude given the raw respiration signal and its
    extrema.

    Parameters
    ----------
    rsp_signal : list, array or Series
        The raw respiration channel (as measured, for instance, by a
        respiration belt).
    extrema : DataFrame, dict
        The extrema (i.e., inhalation peaks and exhalation troughs) found in
        the raw respiration channel. Must be one of the data containers
        returned by `rsp_findpeaks()`.
    desired_length : int
        By default, the returned respiratory amplitude has the same number of
        elements as inhalation peaks in `extrema`. If set to an integer, the
        returned amplitude will be interpolated between inhalation peaks over
        `desired_length` samples. Has not effect if a DataFrame is passed in as
        the `extrema` argument. In the latter case the respiratory amplitude
        will be interpolated over the entire duration of `rsp_signal`.

    Returns
    -------
    array
        A vector containing the respiratory amplitude.

    See Also
    --------
    rsp_clean, rsp_findpeaks, rsp_rate, rsp_process, rsp_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration=90, respiratory_rate=15)
    >>> cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    >>> extrema, info = nk.rsp_findpeaks(cleaned)
    >>>
    >>> amplitude = nk.rsp_amplitude(rsp, extrema)
    """
    # Sanity checks
    error_msg0 = ("NeuroKit error: Please provide one of the containers "
                  "returned by `rsp_findpeaks()` as `extrema` argument.")
    error_msg1 = ("NeuroKit error: Please provide one of the containers "
                  "returned by `rsp_findpeaks()` as `extrema` argument and do "
                  "not modify its content.")
    if isinstance(extrema, dict):
        try:
            peaks = extrema["RSP_Peaks"]
            troughs = extrema["RSP_Troughs"]
        except (TypeError, KeyError):
            raise TypeError(error_msg0)
    elif isinstance(extrema, pd.DataFrame):
        try:
            desired_length = extrema["RSP_Peaks"].size
            peaks = np.where(extrema["RSP_Peaks"] == 1)[0]
            troughs = np.where(extrema["RSP_Troughs"] == 1)[0]
        except (TypeError, KeyError):
            raise TypeError(error_msg0)
    else:
        raise TypeError(error_msg0)

    # To consistenty calculate amplitude, peaks and troughs must have the same
    # number of elements, and the first trough must precede the first peak.
    if peaks.size != troughs.size:
        raise TypeError(error_msg1)
    if peaks[0] <= troughs[0]:
        raise TypeError(error_msg1)


    # Determine length of final signal to return.
    if desired_length is None:
        desired_length = peaks.size

    # Calculate amplitude in units of the raw signal, based on vertical
    # difference of each peak to the preceding trough.
    amplitude = np.abs(rsp_signal[peaks] - rsp_signal[troughs])

    # Interpolate amplitude to desired_length samples.
    amplitude = signal_interpolate(amplitude, x_axis=peaks,
                                   desired_length=desired_length)

    return amplitude

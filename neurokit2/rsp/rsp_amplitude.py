# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_interpolate
from ..signal.signal_formatpeaks import _signal_formatpeaks


def rsp_amplitude(peaks, troughs=None, desired_length=None):
    """Compute respiratory amplitude.

    Compute respiratory amplitude given the raw respiration signal and its
    extrema.

    Parameters
    ----------
    peaks, troughs : list, array, DataFrame, Series or dict
        The samples at which the inhalation peaks occur. If a dict or a
        DataFrame is passed, it is assumed that these containers were obtained
        with `rsp_findpeaks()`.
    desired_length : int
        By default, the returned respiration rate has the same number of
        elements as `peaks`. If set to an integer, the returned rate will be
        interpolated between `peaks` over `desired_length` samples. Has no
        effect if a DataFrame is passed in as the `peaks` argument.

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
    # Format input.
    original_input = peaks
    peaks, desired_length = _signal_formatpeaks(original_input, desired_length, key="Peaks")
    if troughs is None:
        troughs, _ = _signal_formatpeaks(original_input, desired_length, key="Troughs")


    # To consistenty calculate amplitude, peaks and troughs must have the same
    # number of elements, and the first trough must precede the first peak.
    if (peaks.size != troughs.size) or (peaks[0] <= troughs[0]):
        raise TypeError("NeuroKit error: Please provide one of the containers ",
                        "returned by `rsp_findpeaks()` as `extrema` argument and do ",
                        "not modify its content.")


    # Calculate amplitude in units of the raw signal, based on vertical
    # difference of each peak to the preceding trough.
    amplitude = np.abs(peaks - troughs)

    # Interpolate amplitude to desired_length samples.
    amplitude = signal_interpolate(amplitude, x_axis=peaks,
                                   desired_length=desired_length)

    return amplitude

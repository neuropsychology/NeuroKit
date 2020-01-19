# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal.signal_from_indices import _signals_from_peakinfo
from ..signal.signal_rate import signal_rate
from ..signal.signal_formatpeaks import _signal_formatpeaks
from ..signal import signal_interpolate
from ..signal import signal_smooth





def rsp_fixpeaks(peaks, troughs=None, desired_length=None):
    """Correct and format RSP peaks.

    Compute respiration rate with the specified method.

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
    signals : DataFrame
        A DataFrame of same length as the input signal in which occurences of
        inhalation peaks and exhalation troughs are marked as "1" in lists of
        zeros with the same length as `rsp_cleaned`. Accessible with the keys
        "RSP_Peaks" and "RSP_Troughs" respectively.
    info : dict
        A dictionary containing additional information, in this case the
        samples at which inhalation peaks and exhalation troughs occur,
        accessible with the keys "RSP_Peaks", and "RSP_Troughs", respectively.

    See Also
    --------
    rsp_clean, rsp_findpeaks, rsp_amplitude, rsp_process, rsp_plot

    Examples
    --------
    >>>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration=30, respiratory_rate=15)
    >>> cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    >>> info = nk.rsp_findpeaks(cleaned)
    >>> peaks_signal, info = nk.rsp_fixpeaks(info, desired_length=len(cleaned))
    >>>
    >>> data = pd.concat([pd.Series(cleaned), peaks_signal], axis=1)
    >>> nk.signal_plot(data)
    """
    # Format input.
    peaks, troughs, desired_length = _rsp_fixpeaks_retrieve(peaks, troughs, desired_length)

    # Do whatever fixing is required (nothing for now)

    # Prepare output
    info = {"RSP_Peaks": peaks,
            "RSP_Troughs": troughs}

    signals = _signals_from_peakinfo(info, peak_indices=info["RSP_Peaks"], length=desired_length)

    return signals, info


# =============================================================================
# Internals
# =============================================================================
def _rsp_fixpeaks_retrieve(peaks, troughs=None, desired_length=None):
    # Format input.
    original_input = peaks
    peaks, desired_length = _signal_formatpeaks(original_input, desired_length, key="Peaks")
    if troughs is None:
        troughs, _ = _signal_formatpeaks(original_input, desired_length, key="Troughs")
    return peaks, troughs, desired_length

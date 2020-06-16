# -*- coding: utf-8 -*-

import numpy as np

from ..signal import signal_interpolate
from .rsp_fixpeaks import _rsp_fixpeaks_retrieve


def rsp_amplitude(rsp_cleaned, peaks, troughs=None, interpolation_method="monotone_cubic"):
    """Compute respiratory amplitude.

    Compute respiratory amplitude given the raw respiration signal and its extrema.

    Parameters
    ----------
    rsp_cleaned : Union[list, np.array, pd.Series]
        The cleaned respiration channel as returned by `rsp_clean()`.
    peaks : list or array or DataFrame or Series or dict
        The samples at which the inhalation peaks occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with `rsp_findpeaks()`.
    troughs : list or array or DataFrame or Series or dict
        The samples at which the inhalation troughs occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with `rsp_findpeaks()`.
    interpolation_method : str
        Method used to interpolate the amplitude between peaks. See `signal_interpolate()`. 'monotone_cubic' is chosen
        as the default interpolation method since it ensures monotone interpolation between data points
        (i.e., it prevents physiologically implausible "overshoots" or "undershoots" in the y-direction).
        In contrast, the widely used cubic spline interpolation does not ensure monotonicity.


    Returns
    -------
    array
        A vector containing the respiratory amplitude.

    See Also
    --------
    rsp_clean, rsp_peaks, signal_rate, rsp_process, rsp_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>> import pandas as pd
    >>>
    >>> rsp = nk.rsp_simulate(duration=90, respiratory_rate=15)
    >>> cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    >>> info, signals = nk.rsp_peaks(cleaned)
    >>>
    >>> amplitude = nk.rsp_amplitude(cleaned, signals)
    >>> fig = nk.signal_plot(pd.DataFrame({"RSP": rsp, "Amplitude": amplitude}), subplots=True)
    >>> fig #doctest: +SKIP

    """
    # Format input.
    peaks, troughs = _rsp_fixpeaks_retrieve(peaks, troughs)

    # To consistenty calculate amplitude, peaks and troughs must have the same
    # number of elements, and the first trough must precede the first peak.
    if (peaks.size != troughs.size) or (peaks[0] <= troughs[0]):
        raise TypeError(
            "NeuroKit error: Please provide one of the containers ",
            "returned by `rsp_findpeaks()` as `extrema` argument and do ",
            "not modify its content.",
        )

    # Calculate amplitude in units of the raw signal, based on vertical
    # difference of each peak to the preceding trough.
    amplitude = rsp_cleaned[peaks] - rsp_cleaned[troughs]

    # Interpolate amplitude to length of rsp_cleaned.
    amplitude = signal_interpolate(peaks, amplitude, x_new=np.arange(len(rsp_cleaned)), method=interpolation_method)

    return amplitude

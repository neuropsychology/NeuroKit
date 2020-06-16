# -*- coding: utf-8 -*-

from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize


def rsp_fixpeaks(peaks, troughs=None):
    """Correct RSP peaks.

    Low-level function used by `rsp_peaks()` to correct the peaks found by `rsp_findpeaks()`.
    Doesn't do anything for now for RSP. See `rsp_peaks()` for details.

    Parameters
    ----------
    peaks : list or array or DataFrame or Series or dict
        The samples at which the inhalation peaks occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with `rsp_findpeaks()`.
    troughs : list or array or DataFrame or Series or dict
        The samples at which the inhalation troughs occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with `rsp_findpeaks()`.

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the samples at which inhalation
        peaks and exhalation troughs occur, accessible with the keys "RSP_Peaks", and "RSP_Troughs", respectively.

    See Also
    --------
    rsp_clean, rsp_findpeaks, rsp_peaks, rsp_amplitude, rsp_process, rsp_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration=30, respiratory_rate=15)
    >>> cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    >>> info = nk.rsp_findpeaks(cleaned)
    >>> info = nk.rsp_fixpeaks(info)
    >>> fig = nk.events_plot([info["RSP_Peaks"], info["RSP_Troughs"]], cleaned)
    >>> fig #doctest: +SKIP

    """
    # Format input.
    peaks, troughs = _rsp_fixpeaks_retrieve(peaks, troughs)

    # Do whatever fixing is required (nothing for now)

    # Prepare output
    info = {"RSP_Peaks": peaks, "RSP_Troughs": troughs}

    return info


# =============================================================================
# Internals
# =============================================================================
def _rsp_fixpeaks_retrieve(peaks, troughs=None):
    # Format input.
    original_input = peaks
    peaks = _signal_formatpeaks_sanitize(original_input, key="Peaks")
    if troughs is None:
        troughs = _signal_formatpeaks_sanitize(original_input, key="Troughs")
    return peaks, troughs

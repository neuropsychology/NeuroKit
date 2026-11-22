import numpy as np
import pandas as pd

from ..signal import signal_filter, signal_interpolate, signal_rate, signal_resample
from .rsp_peaks import rsp_peaks


def rsp_rate(
    rsp_cleaned,
    troughs=None,
    sampling_rate=1000,
    window=10,
    hop_size=1,
    method="trough",
    peak_method="khodadad2018",
    interpolation_method="monotone_cubic",
):
    """**Find respiration rate**

    Parameters
    ----------
    rsp_cleaned : Union[list, np.array, pd.Series]
        The cleaned respiration channel as returned by :func:`.rsp_clean`.
    troughs : Union[list, dict, np.array, pd.Series, pd.DataFrame]
        The respiration troughs (inhalation onsets) as returned by :func:`.rsp_peaks`.
        If None (default), inhalation onsets will be automatically identified from the
        :func:`.rsp_clean` signal.
    sampling_rate : int
        The sampling frequency of :func:`.rsp_cleaned` (in Hz, i.e., samples/second).
    window : int
        The duration of the sliding window (in second). Default to 10 seconds.
    hop_size : int
        The number of samples between each successive window. Default to 1 sample.
    method : str
        Method can either be ``"trough"`` or ``"xcorr"``. In ``"trough"`` method, respiratory rate
        is calculated from the periods between successive inspirations (i.e., inhalation onsets/
        troughs). In ``"xcorr"`` method, cross-correlations between the changes in respiration with
        a bank of sinusoids of different frequencies are calculated to identify the principal
        frequency of oscillation.
    peak_method : str
        Method to identify successive respiratory inspirations, only relevant if method is
        ``"trough"``. Can be one of ``"khodadad2018"`` (default) or ``"biosppy"``.
    interpolation_method : str
        Method used to interpolate the rate between inhalation onsets.
        See :func:`.signal_interpolate`. ``"monotone_cubic"`` is chosen as the default
        interpolation method since it ensures monotone interpolation between data points (i.e., it
        prevents physiologically implausible "overshoots" or "undershoots" in the
        y-direction). In contrast, the widely used cubic spline interpolation does not ensure
        monotonicity.

    Return
    ------
    rsp_rate : np.ndarray
        Instantenous respiration rate.

    Example
    -------
    .. ipython:: python

      import neurokit2 as nk
      rsp_signal = nk.data("rsp_1000hz")

      rsp_cleaned = nk.rsp_clean(rsp_signal, sampling_rate=1000)
      rsp_rate_onsets = nk.rsp_rate(rsp_cleaned, sampling_rate=1000, method="trough")
      rsp_rate_xcorr = nk.rsp_rate(rsp_cleaned, sampling_rate=1000, method="xcorr")

    """

    if method.lower() in ["period", "peak", "peaks", "trough", "troughs", "signal_rate"]:
        if troughs is None:
            _, troughs = rsp_peaks(rsp_cleaned, sampling_rate=sampling_rate, method=peak_method)
        if isinstance(troughs, tuple):
            # Handle (signals, info) tuple returned directly by rsp_peaks()
            troughs = troughs[1]
        if isinstance(troughs, (pd.DataFrame, dict)):
            troughs = troughs["RSP_Troughs"]
        troughs = np.asarray(troughs)
        if len(troughs) > 2 and len(troughs) == len(rsp_cleaned) and np.all(np.isin(troughs, [0, 1])):
            troughs = np.where(troughs == 1)[0]
        rate = signal_rate(
            troughs,
            sampling_rate=sampling_rate,
            desired_length=len(rsp_cleaned),
            interpolation_method=interpolation_method,
        )

    elif method.lower() in ["cross-correlation", "xcorr"]:
        rate = _rsp_rate_xcorr(
            rsp_cleaned,
            sampling_rate=sampling_rate,
            window=window,
            hop_size=hop_size,
            interpolation_method=interpolation_method,
        )

    else:
        raise ValueError("NeuroKit error: rsp_rate(): 'method' should be one of 'trough', or 'cross-correlation'.")

    return rate


# =============================================================================
# Cross-correlation method
# =============================================================================


def _rsp_rate_xcorr(rsp_cleaned, sampling_rate=1000, window=10, hop_size=1, interpolation_method="monotone_cubic"):
    N = len(rsp_cleaned)
    if N == 0:
        return np.array([], dtype=float)

    # Downsample data to 10 Hz (sufficient for up to 30 breaths/min = 0.5 Hz)
    desired_sampling_rate = 10
    if sampling_rate > desired_sampling_rate:
        rsp = signal_resample(rsp_cleaned, sampling_rate=sampling_rate, desired_sampling_rate=desired_sampling_rate)
        effective_sr = desired_sampling_rate
    else:
        rsp = np.asarray(rsp_cleaned, dtype=float)
        effective_sr = sampling_rate

    if len(rsp) < 2:
        return np.full(N, np.nan)

    window_length = max(2, min(len(rsp), int(round(effective_sr * window))))
    hop_ds = max(1, int(round(hop_size * effective_sr / sampling_rate)))

    starts = np.arange(0, len(rsp) - window_length + 1, hop_ds)
    all_windows = np.lib.stride_tricks.sliding_window_view(rsp, window_length)[starts]

    # Center and normalize each window
    win_centered = all_windows - all_windows.mean(axis=1, keepdims=True)
    win_norms = np.linalg.norm(win_centered, axis=1, keepdims=True)
    valid = win_norms[:, 0] > 0
    if not np.any(valid):
        return np.full(N, np.nan)

    win_unit = np.zeros_like(win_centered)
    win_unit[valid] = win_centered[valid] / win_norms[valid]

    # Build orthonormal quadrature (sin, cos) basis across candidate frequencies (5 to 30 bpm)
    frequencies = np.arange(5 / 60, 30.25 / 60, 0.25 / 60)
    t = np.arange(window_length) / effective_sr
    angles = 2 * np.pi * frequencies[:, None] * t[None, :]

    sin_centered = np.sin(angles)
    sin_centered -= sin_centered.mean(axis=1, keepdims=True)
    e1 = sin_centered / np.linalg.norm(sin_centered, axis=1, keepdims=True)

    cos_centered = np.cos(angles)
    cos_centered -= cos_centered.mean(axis=1, keepdims=True)
    cos_orth = cos_centered - np.sum(cos_centered * e1, axis=1, keepdims=True) * e1
    e2 = cos_orth / np.linalg.norm(cos_orth, axis=1, keepdims=True)

    # Phase-invariant correlation amplitude for every window and candidate frequency
    xcorr = np.hypot(win_unit @ e1.T, win_unit @ e2.T)
    max_freqs = frequencies[np.argmax(xcorr, axis=1)]

    # Interpolate valid window centers back to original sample grid
    centers = (starts[valid] + (window_length - 1) / 2.0) * (sampling_rate / effective_sr)
    rsp_rate = signal_interpolate(centers, max_freqs, x_new=np.arange(N), method=interpolation_method)

    # Smooth instantaneous rate when signal is long enough for order-4 filtfilt and Nyquist allows 0.1 Hz
    if N > 15 and sampling_rate > 0.2 and np.ptp(rsp_rate) > 0:
        rsp_rate = signal_filter(rsp_rate, highcut=0.1, order=4, sampling_rate=sampling_rate)

    # Convert Hz to breaths per minute
    rsp_rate = np.multiply(rsp_rate, 60)

    return np.asarray(rsp_rate, dtype=float)

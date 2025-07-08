# -*- coding: utf-8 -*-

import numpy as np

from ..signal import signal_interpolate
from .rsp_fixpeaks import _rsp_fixpeaks_retrieve


def rsp_amplitude(rsp_cleaned, peaks, troughs=None, method="standard", interpolation_method="monotone_cubic"):
    """**Compute respiratory amplitude**

    Compute respiratory amplitude given the raw respiration signal and its extrema. The
    **standard** method computes the amplitude as the difference between a peak and its preceding
    trough, while the **prepost** method computes the amplitude as the average of the differences
    of peak to its preceding and succeeding troughs (Cole, 2019).

    Parameters
    ----------
    rsp_cleaned : Union[list, np.array, pd.Series]
        The cleaned respiration channel as returned by :func:`.rsp_clean`.
    peaks : list or array or DataFrame or Series or dict
        The samples at which the respiration peaks (exhalation onsets) occur. If a dict or a
        DataFrame is passed, it is assumed that these containers were obtained
        with :func:`rsp_findpeaks`.
    troughs : list or array or DataFrame or Series or dict
        The samples at which the respiration troughs (inhalation onsets) occur. If a dict or a
        is passed, it is assumed that these containers were obtained with :func:`.rsp_findpeaks`.
    method : str
        The method to use to compute the amplitude. Can be ``"standard"`` or ``"prepost"``.
    interpolation_method : str
        Method used to interpolate the amplitude between peaks. See :func:`.signal_interpolate`.
        ``"monotone_cubic"`` is chosen as the default interpolation method since it ensures monotone
        interpolation between data point (i.e., it prevents physiologically implausible "overshoots"
        or "undershoots" in the y-direction). In contrast, the widely used cubic spline
        'interpolation does not ensure monotonicity.

    Returns
    -------
    array
        A vector containing the respiratory amplitude.

    See Also
    --------
    rsp_clean, rsp_peaks, signal_rate, rsp_process, rsp_plot, rsp_symmetry

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      rsp = nk.rsp_simulate(duration=90, respiratory_rate=15)
      cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
      peak_signals, info = nk.rsp_peaks(cleaned)

      amplitude = nk.rsp_amplitude(cleaned, peak_signals)
      @savefig p_rsp_amp1.png scale=100%
      fig = nk.signal_plot([rsp, amplitude], labels=["RSP", "Amplitude"], subplots=True)
      @suppress
      plt.close()

    .. ipython:: python

      amp2 = nk.rsp_amplitude(cleaned, peak_signals, method="prepost")
      @savefig p_rsp_amp2.png scale=100%
      fig = nk.signal_plot([amplitude, amp2], labels=["Standard", "Prepost"])
      @suppress
      plt.close()

    References
    ----------
    * Cole, S., & Voytek, B. (2019). Cycle-by-cycle analysis of neural oscillations. Journal of
      neurophysiology, 122(2), 849-861.

    """
    # Make sure `rsp_cleaned` is a np.array
    rsp_cleaned = np.array(rsp_cleaned)

    # Format input.
    peaks, troughs = _rsp_fixpeaks_retrieve(peaks, troughs)

    # To consistently calculate amplitude, peaks and troughs must have the same
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

    # The above is the standard amplitude (each peak height to the preceding trough).
    if method in ["prepost"]:
        # Alternative amplitude calculation that corresponds to the average of
        # the peak height to the preceding and following troughs.
        # https://twitter.com/bradleyvoytek/status/1591495571269124096/photo/1
        # (Note that it cannot be done for the last peak)
        amplitude[0:-1] += rsp_cleaned[peaks[0:-1]] - rsp_cleaned[troughs[1::]]
        amplitude[0:-1] /= 2

    # Interpolate amplitude to length of rsp_cleaned.
    if len(peaks) == 1:
        amplitude = np.full(rsp_cleaned.shape, amplitude[0])
    else:
        amplitude = signal_interpolate(peaks, amplitude, x_new=np.arange(len(rsp_cleaned)), method=interpolation_method)

    return amplitude

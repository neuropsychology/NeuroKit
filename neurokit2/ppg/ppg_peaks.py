import matplotlib.pyplot as plt
import numpy as np

from ..signal import signal_formatpeaks
from .ppg_findpeaks import ppg_findpeaks


def ppg_peaks(ppg_cleaned, sampling_rate=1000, method="elgendi", show=False, **kwargs):
    """**Find systolic peaks in a photoplethysmogram (PPG) signal**

    Find the peaks in an PPG signal using the specified method. You can pass an unfiltered PPG
    signals as input, but typically a filtered PPG (cleaned using ``ppg_clean()``) will result in
    better results.

    .. note::

      Please help us improve the methods' documentation and features.


    Parameters
    ----------
    ppg_cleaned : Union[list, np.array, pd.Series]
        The cleaned PPG channel as returned by ``ppg_clean()``.
    sampling_rate : int
        The sampling frequency of ``ppg_cleaned`` (in Hz, i.e., samples/second). Defaults to 1000.
    method : str
        The processing pipeline to apply. Can be one of ``"elgendi"``, ``"bishop"``. The default is
        ``"elgendi"``.
    show : bool
        If ``True``, will show a plot of the signal with peaks. Defaults to ``False``.
    **kwargs
        Additional keyword arguments, usually specific for each method.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as the input signal in which occurrences of R-peaks marked as
        ``1`` in a list of zeros with the same length as ``ppg_cleaned``. Accessible with the keys
        ``"PPG_Peaks"``.
    info : dict
        A dictionary containing additional information, in this case the samples at which R-peaks
        occur, accessible with the key ``"PPG_Peaks"``, as well as the signals' sampling rate,
        accessible with the key ``"sampling_rate"``.

    See Also
    --------
    ppg_clean, ppg_segment, .signal_fixpeaks

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      import matplotlib.pyplot as plt

      ppg = nk.ppg_simulate(heart_rate=75, duration=20, sampling_rate=50)
      ppg_clean = nk.ppg_clean(ppg, sampling_rate=50)

      # Default method (Elgendi et al., 2013)
      @savefig p_ppg_peaks1.png scale=100%
      peaks, info = nk.ppg_peaks(ppg_clean, sampling_rate=100, method="elgendi", show=True)
      @suppress
      plt.close()
      peaks_idx = info["PPG_Peaks"]

      # Method by Bishop et al., (2018)
      @savefig p_ppg_peaks2.png scale=100%
      peaks = nk.ppg_peaks(ppg_clean, sampling_rate=100, method="bishop", show=True)
      @suppress
      plt.close()

    References
    ----------
    * Elgendi, M., Norton, I., Brearley, M., Abbott, D., & Schuurmans, D. (2013). Systolic peak
      detection in acceleration photoplethysmograms measured from emergency responders in tropical
      conditions. PloS one, 8(10), e76585.
    * Bishop, S. M., & Ercole, A. (2018). Multi-scale peak and trough detection optimised for
      periodic and quasi-periodic neuroscience data. In Intracranial Pressure & Neuromonitoring XVI
      (pp. 189-195). Springer International Publishing.

    """
    peaks = ppg_findpeaks(
        ppg_cleaned, sampling_rate=sampling_rate, method=method, show=False, **kwargs
    )

    instant_peaks = signal_formatpeaks(
        peaks, desired_length=len(ppg_cleaned), peak_indices=peaks
    )
    signals = instant_peaks
    info = peaks
    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    if show is True:
        _ppg_peaks_plot(ppg_cleaned, info, sampling_rate)

    return signals, info


# =============================================================================
# Internals
# =============================================================================
def _ppg_peaks_plot(
    ppg_cleaned,
    info=None,
    sampling_rate=1000,
    raw=None,
    ax=None,
):
    x_axis = np.linspace(0, len(ppg_cleaned) / sampling_rate, len(ppg_cleaned))

    # Prepare plot
    if ax is None:
        _, ax = plt.subplots()

    ax.set_xlabel("Time (seconds)")
    ax.set_title("PPG signal and peaks")

    # Raw Signal ---------------------------------------------------------------
    if raw is not None:
        ax.plot(x_axis, raw, color="#B0BEC5", label="Raw signal", zorder=1)
        label_clean = "Cleaned signal"
    else:
        label_clean = "Signal"

    # Peaks -------------------------------------------------------------------
    ax.scatter(
        x_axis[info["PPG_Peaks"]],
        ppg_cleaned[info["PPG_Peaks"]],
        color="#FFC107",
        label="Systolic peaks",
        zorder=2,
    )

    # Clean Signal ------------------------------------------------------------
    ax.plot(
        x_axis,
        ppg_cleaned,
        color="#9C27B0",
        label=label_clean,
        zorder=3,
        linewidth=1,
    )

    # Optimize legend
    ax.legend(loc="upper right")

    return ax

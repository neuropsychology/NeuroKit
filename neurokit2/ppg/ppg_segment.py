# - * - coding: utf-8 - * -
import numpy as np

from ..ecg.ecg_segment import _ecg_segment_plot, _ecg_segment_window
from ..epochs import epochs_create
from .ppg_peaks import ppg_peaks


def ppg_segment(ppg_cleaned, peaks=None, sampling_rate=1000, show=False, **kwargs):
    """**Segment an PPG signal into single heartbeats**

    Segment a PPG signal into single heartbeats. Convenient for visualizing all the heart beats.

    Parameters
    ----------
    ppg_cleaned : Union[list, np.array, pd.Series]
        The cleaned PPG channel as returned by ``ppg_clean()``.
    peaks : dict
        The samples at which the R-peaks occur. Dict returned by ``ppg_peaks()``. Defaults to ``None``.
    sampling_rate : int
        The sampling frequency of ``ppg_cleaned`` (in Hz, i.e., samples/second). Defaults to 1000.
    show : bool
        If ``True``, will return a plot of heartbeats. Defaults to ``False``.
    **kwargs
        Other arguments to be passed.

    Returns
    -------
    dict
        A dict containing DataFrames for all segmented heartbeats.

    See Also
    --------
    ppg_clean, ppg_plot

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      ppg = nk.ppg_simulate(duration=30, sampling_rate=100, heart_rate=80)
      @savefig p_ppg_segment.png scale=100%
      ppg_epochs = nk.ppg_segment(ppg, sampling_rate=100, show=True)
      @suppress
      plt.close()

    """
    # Sanitize inputs
    if peaks is None:
        _, peaks = ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate)
        peaks = peaks["PPG_Peaks"]

    if len(ppg_cleaned) < sampling_rate * 4:
        raise ValueError("The data length is too small to be segmented.")

    epochs_start, epochs_end, average_hr = _ecg_segment_window(
        rpeaks=peaks,
        sampling_rate=sampling_rate,
        desired_length=len(ppg_cleaned),
        ratio_pre=0.3,
    )
    heartbeats = epochs_create(
        ppg_cleaned,
        peaks,
        sampling_rate=sampling_rate,
        epochs_start=epochs_start,
        epochs_end=epochs_end,
    )

    # pad last heartbeat with nan so that segments are equal length
    last_heartbeat_key = str(np.max(np.array(list(heartbeats.keys()), dtype=int)))
    after_last_index = heartbeats[last_heartbeat_key]["Index"] >= len(ppg_cleaned)
    heartbeats[last_heartbeat_key].loc[after_last_index, "Signal"] = np.nan

    if show is not False:
        ax = _ecg_segment_plot(
            heartbeats, heartrate=average_hr, ytitle="PPG", color="#E91E63", **kwargs
        )
    if show == "return":
        return ax

    return heartbeats

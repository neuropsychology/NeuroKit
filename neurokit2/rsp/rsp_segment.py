# - * - coding: utf-8 - * -
import numpy as np

from ..ecg.ecg_segment import _ecg_segment_plot
from .rsp_peaks import rsp_peaks
from ..signal import signal_cyclesegment


def rsp_segment(rsp_cleaned, peaks=None, sampling_rate=1000, show=False, **kwargs):
    """**Segment a respiratory signal into individual breaths**

    Segment a respiratory signal into individual breaths. Convenient for visualizing all the breaths.

    Parameters
    ----------
    rsp_cleaned : Union[list, np.array, pd.Series]
        The cleaned respiratory channel as returned by ``rsp_clean()``.
    peaks : dict
        The samples at which the peaks (exhalation onsets) occur. Dict returned by ``rsp_peaks()``. Defaults to ``None``.
    sampling_rate : int
        The sampling frequency of ``rsp_cleaned`` (in Hz, i.e., samples/second). Defaults to 1000.
    show : bool
        If ``True``, will return a plot of breaths. Defaults to ``False``.
    **kwargs
        Other arguments to be passed.

    Returns
    -------
    dict
        A dict containing DataFrames for all segmented breaths.

    See Also
    --------
    rsp_clean, rsp_peaks

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      sampling_rate=100
      rsp = nk.rsp_simulate(duration=30, sampling_rate=sampling_rate, method="breathmetrics")
      @savefig p_rsp_segment.png scale=100%
      rsp_epochs = nk.rsp_segment(rsp, sampling_rate=sampling_rate, show=True)
      @suppress
      plt.close()

    """
    # Sanitize inputs
    if peaks is None:
        _, peaks = rsp_peaks(rsp_cleaned, sampling_rate=sampling_rate)
        peaks = peaks["RSP_Peaks"]

    if len(rsp_cleaned) < sampling_rate * 10:
        raise ValueError("The data length is too small to be segmented.")

    breaths, average_rr = signal_cyclesegment(rsp_cleaned, peaks, sampling_rate=sampling_rate)

    if show is not False:
        ax = _ecg_segment_plot(
            breaths, heartrate=average_rr, ytitle="RSP", color="#E91E63", **kwargs
        )
    if show == "return":
        return ax

    return breaths

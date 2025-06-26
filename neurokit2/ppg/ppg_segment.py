# - * - coding: utf-8 - * -
from .ppg_peaks import ppg_peaks
from ..signal import signal_cyclesegment


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
    
    heartbeats, average_hr = signal_cyclesegment(ppg_cleaned, peaks, ratio_pre=0.3, sampling_rate=sampling_rate, show=show, signal_name="ppg")

    return heartbeats

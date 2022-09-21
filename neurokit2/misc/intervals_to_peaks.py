import numpy as np
from .find_successive_intervals import find_successive_intervals


def intervals_to_peaks(intervals, intervals_time=None, sampling_rate=1000):
    """Convenience function to convert intervals to peaks, such as from R-R intervals to R-peaks of an ECG signal.

    This can be useful if you do not have raw peak indices and have only
    interval data such as breath-to-breath (BBI) or rpeak-to-rpeak (RRI) intervals.

    Parameters
    ----------
    intervals : list or array
        List or numpy array of intervals, in milliseconds.
    intervals_time : list or array, optional
        List or numpy array of timestamps corresponding to intervals, in seconds.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks occur.

    Returns
    -------
    array
        An array of integer values indicating the peak indices,
        with the first peak occurring at sample point 0.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk
      ibi = [500, 400, 700, 500, 300, 800, 500]
      peaks = nk.intervals_to_peaks(ibi)
      @savefig p_intervals_to_peaks.png scale=100%
      hrv_indices = nk.hrv_time(peaks, sampling_rate=100, show=True)
      @suppress
      plt.close()
      hrv_indices

    """
    if intervals is None:
        return None

    if intervals_time is None:
        intervals_time = np.nancumsum(intervals) / 1000

    intervals_time = intervals_time[np.isfinite(intervals)]
    intervals = intervals[np.isfinite(intervals)]

    non_successive_indices = np.arange(1, len(intervals_time))[np.invert(
        find_successive_intervals(intervals, intervals_time))]

    to_insert_indices = np.concatenate(
        (np.array([0]), non_successive_indices))

    times_to_insert = intervals_time[to_insert_indices] - \
        intervals[to_insert_indices]/1000

    peaks_time = np.sort(np.concatenate((intervals_time, times_to_insert)))
    peaks = peaks_time*sampling_rate

    return np.array([int(np.round(i)) for i in peaks])

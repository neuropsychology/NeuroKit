import numpy as np

from .intervals_utils import _intervals_sanitize, _intervals_successive


def intervals_to_peaks(intervals, intervals_time=None, sampling_rate=1000):
    """**Convert intervals to peaks**

    Convenience function to convert intervals to peaks, such as from R-R intervals to R-peaks of an
    ECG signal. This can be useful if you do not have raw peak indices and have only interval data
    such as breath-to-breath (BBI) or rpeak-to-rpeak (RRI) intervals.

    Parameters
    ----------
    intervals : list or array
        List or numpy array of intervals, in milliseconds.
    intervals_time : list or array, optional
        List or numpy array of timestamps corresponding to intervals, in seconds.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous signal in which the peaks occur.

    Returns
    -------
    np.ndarray
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

    intervals, intervals_time, intervals_missing = _intervals_sanitize(
        intervals, intervals_time=intervals_time, remove_missing=True
    )

    if intervals_missing:
        # Check for non successive intervals in case of missing data
        non_successive_indices = np.arange(1, len(intervals_time))[
            np.invert(_intervals_successive(intervals, intervals_time))
        ]
    else:
        non_successive_indices = np.array([]).astype(int)
    # The number of peaks should be the number of intervals
    # plus one extra at the beginning of each group of successive intervals
    # (with no missing data there should be N_intervals + 1 peaks)
    to_insert_indices = np.concatenate((np.array([0]), non_successive_indices))

    times_to_insert = intervals_time[to_insert_indices] - intervals[to_insert_indices] / 1000

    peaks_time = np.sort(np.concatenate((intervals_time, times_to_insert)))
    # convert seconds to sample indices
    peaks = peaks_time * sampling_rate

    return np.array([int(np.round(i)) for i in peaks])

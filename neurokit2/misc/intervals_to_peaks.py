import numpy as np


def intervals_to_peaks(intervals):
    """Convenience function to convert intervals to peaks,
    such as from R-R intervals to R-peaks of an ECG signal.

    This can be useful if you do not have raw peak indices and have only
    interval data such as breath-to-breath (BBI) or rpeak-to-rpeak (RRI) intervals.

    Parameters
    ----------
    intervals : list or array
        List or numpy array of intervals.

    Returns
    -------
    array
        An array of integer values indicating the peak indices,
        with the first peak occurring at sample point 0.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>> ibi = [500, 400, 700, 500, 300, 800, 500]
    >>> peaks = nk.intervals_to_peaks(ibi)
    >>> hrv_indices = nk.hrv_time(peaks, sampling_rate=100, show=True)

    """
    peaks = np.append([0], np.cumsum(intervals))

    return np.array([int(i) for i in peaks])

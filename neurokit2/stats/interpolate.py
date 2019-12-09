# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d


def interp_stats(peaks, stats, nsamp):
    """Interpolate statistics over the entire duration of the signal.

    Parameters
    ----------
    peaks : 1d array
        R-peaks in ECG, breathing peaks in breathing signal.
    stats : 1d array
        Any statistic that is calculated on on the difference of peak t to
        peak t-1 and is assigned to peak t. Must have the same number of
        elements as peaks.
    nsamp : int
        Desired number of samples in the returned time series.

    Returns
    -------
    statsintp : 1d array
        Values in stats interpolated over nsamp samples.

    """
    # Samples up until the first peak as well as from last peak to end of
    # signal are set to the value of the first and last element of stats
    # respectively. Linear (2nd order) interpolation is chosen since higher
    # order interpolation can lead to biologically implausible values and
    # erratic fluctuations.
    f = interp1d(np.ravel(peaks), stats, kind='slinear',
                 bounds_error=False, fill_value=([stats[0]], [stats[-1]]))
    samples = np.arange(0, nsamp)
    statsintp = f(samples)

    return statsintp

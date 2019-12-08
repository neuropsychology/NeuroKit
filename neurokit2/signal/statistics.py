# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d


def interp_stats(peaks, stats, nsamp):
    """
    Interpolate descriptive statistics over the entire duration of the
    signal: samples up until first peak as well as from last peak to end of
    signal are set to the value of the first and last element of stats
    respectively.
    Linear (2nd order) interpolation is chosen since cubic (4th order)
    interpolation can lead to biologically implausible interpolated values
    and erratic fluctuations due to overfitting.

    input:
        peaks: R-peaks in ECG, breathing peaks in breathing signal
        stats: any statistic that is calculated on on the difference of peak t
        to peak t-1 and is assigned to peak t. Must have the same number of
        elements as peaks
        nsamp: desired number of sampled in the returned time series

    returns:
        statsintp: stats interpolated over nsamp samples
    """

    f = interp1d(np.ravel(peaks), stats, kind='slinear',
                 bounds_error=False, fill_value=([stats[0]], [stats[-1]]))
    # internally, for consistency in plotting etc., keep original sampling
    # rate
    samples = np.arange(0, nsamp)
    statsintp = f(samples)

    return statsintp

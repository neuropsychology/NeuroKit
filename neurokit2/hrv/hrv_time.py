# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from ..stats import mad, summary_plot
from .hrv_utils import _hrv_get_rri, _hrv_sanitize_input


def hrv_time(peaks, sampling_rate=1000, show=False, **kwargs):
    """Computes time-domain indices of Heart Rate Variability (HRV).

     See references for details.

    Parameters
    ----------
    peaks : dict
        Samples at which cardiac extrema (i.e., R-peaks, systolic peaks) occur.
        Can be a list of indices or the output(s) of other functions such as ecg_peaks,
        ppg_peaks, ecg_process or bio_process.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks occur. Should be at
        least twice as high as the highest frequency in vhf. By default 1000.
    show : bool
        If True, will plot the distribution of R-R intervals.

    Returns
    -------
    DataFrame
        Contains time domain HRV metrics:
        - **MeanNN**: The mean of the RR intervals.
        - **SDNN**: The standard deviation of the RR intervals.
        -**SDANN1**, **SDANN2**, **SDANN5**: The standard deviation of average RR intervals extracted from n-minute segments of
        time series data (1, 2 and 5 by default). Note that these indices require a minimal duration of signal to be computed
        (3, 6 and 15 minutes respectively) and will be silently skipped if the data provided is too short.
        -**SDNNI1**, **SDNNI2**, **SDNNI5**: The mean of the standard deviations of RR intervals extracted from n-minute
        segments of time series data (1, 2 and 5 by default). Note that these indices require a minimal duration of signal to
        be computed (3, 6 and 15 minutes respectively) and will be silently skipped if the data provided is too short.
        - **RMSSD**: The square root of the mean of the sum of successive differences between
        adjacent RR intervals. It is equivalent (although on another scale) to SD1, and
        therefore it is redundant to report correlations with both (Ciccone, 2017).
        - **SDSD**: The standard deviation of the successive differences between RR intervals.
        - **CVNN**: The standard deviation of the RR intervals (SDNN) divided by the mean of the RR
        intervals (MeanNN).
        - **CVSD**: The root mean square of the sum of successive differences (RMSSD) divided by the
        mean of the RR intervals (MeanNN).
        - **MedianNN**: The median of the absolute values of the successive differences between RR intervals.
        - **MadNN**: The median absolute deviation of the RR intervals.
        - **HCVNN**: The median absolute deviation of the RR intervals (MadNN) divided by the median
        of the absolute differences of their successive differences (MedianNN).
        - **IQRNN**: The interquartile range (IQR) of the RR intervals.
        - **pNN50**: The proportion of RR intervals greater than 50ms, out of the total number of RR intervals.
        - **pNN20**: The proportion of RR intervals greater than 20ms, out of the total number of RR intervals.
        - **TINN**: A geometrical parameter of the HRV, or more specifically, the baseline width of
        the RR intervals distribution obtained by triangular interpolation, where the error of least
        squares determines the triangle. It is an approximation of the RR interval distribution.
        - **HTI**: The HRV triangular index, measuring the total number of RR intervals divded by the
        height of the RR intervals histogram.

    See Also
    --------
    ecg_peaks, ppg_peaks, hrv_frequency, hrv_summary, hrv_nonlinear

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Download data
    >>> data = nk.data("bio_resting_5min_100hz")
    >>>
    >>> # Find peaks
    >>> peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)
    >>>
    >>> # Compute HRV indices
    >>> hrv = nk.hrv_time(peaks, sampling_rate=100, show=True)

    References
    ----------
    - Ciccone, A. B., Siedlik, J. A., Wecht, J. M., Deckert, J. A., Nguyen, N. D., & Weir, J. P.
    (2017). Reminder: RMSSD and SD1 are identical heart rate variability metrics. Muscle & nerve,
    56(4), 674-678.

    - Stein, P. K. (2002). Assessing heart rate variability from real-world Holter reports. Cardiac
    electrophysiology review, 6(3), 239-244.

    - Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms.
    Frontiers in public health, 5, 258.

    """
    # Sanitize input
    peaks = _hrv_sanitize_input(peaks)
    if isinstance(peaks, tuple):  # Detect actual sampling rate
        peaks, sampling_rate = peaks[0], peaks[1]

    # Compute R-R intervals (also referred to as NN) in milliseconds
    rri = _hrv_get_rri(peaks, sampling_rate=sampling_rate, interpolate=False)
    diff_rri = np.diff(rri)

    out = {}  # Initialize empty container for results

    # Deviation-based
    out["MeanNN"] = np.nanmean(rri)
    out["SDNN"] = np.nanstd(rri, ddof=1)
    for i in [1, 2, 5]:
        out["SDANN" + str(i)] = _sdann(rri, sampling_rate, window=i)
        out["SDNNI" + str(i)] = _sdnni(rri, sampling_rate, window=i)

    # Difference-based
    out["RMSSD"] = np.sqrt(np.mean(diff_rri ** 2))
    out["SDSD"] = np.nanstd(diff_rri, ddof=1)

    # Normalized
    out["CVNN"] = out["SDNN"] / out["MeanNN"]
    out["CVSD"] = out["RMSSD"] / out["MeanNN"]

    # Robust
    out["MedianNN"] = np.nanmedian(rri)
    out["MadNN"] = mad(rri)
    out["MCVNN"] = out["MadNN"] / out["MedianNN"]  # Normalized
    out["IQRNN"] = scipy.stats.iqr(rri)

    # Extreme-based
    nn50 = np.sum(np.abs(diff_rri) > 50)
    nn20 = np.sum(np.abs(diff_rri) > 20)
    out["pNN50"] = nn50 / len(rri) * 100
    out["pNN20"] = nn20 / len(rri) * 100

    # Geometrical domain
    if "binsize" in kwargs:
        binsize = kwargs["binsize"]
    else:
        binsize = (1 / 128) * 1000
    bins = np.arange(0, np.max(rri) + binsize, binsize)
    bar_y, bar_x = np.histogram(rri, bins=bins)
    # HRV Triangular Index
    out["HTI"] = len(rri) / np.max(bar_y)
    # Triangular Interpolation of the NN Interval Histogram
    out["TINN"] = _hrv_TINN(rri, bar_x, bar_y, binsize)

    if show:
        _hrv_time_show(rri, **kwargs)

    out = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("HRV_")
    return out

# =============================================================================
# Utilities
# =============================================================================

def _hrv_time_show(rri, **kwargs):

    fig = summary_plot(rri, **kwargs)
    plt.xlabel("R-R intervals (ms)")
    fig.suptitle("Distribution of R-R intervals")

    return fig

def _sdann(rri, sampling_rate, window=1):

    window_size = window * 60 * 1000  # Convert window in min to ms
    n_windows = int(np.round(np.cumsum(rri)[-1] / window_size))
    if n_windows < 3:
        return np.nan
    rri_cumsum = np.cumsum(rri)
    avg_rri = []
    for i in range(n_windows):
        start = i * window_size
        start_idx = np.where(rri_cumsum >= start)[0][0]
        end_idx = np.where(rri_cumsum < start + window_size)[0][-1]
        avg_rri.append(np.mean(rri[start_idx:end_idx]))
    sdann = np.nanstd(avg_rri, ddof=1)
    return sdann

def _sdnni(rri, sampling_rate, window=1):

    window_size = window * 60 * 1000  # Convert window in min to ms
    n_windows = int(np.round(np.cumsum(rri)[-1] / window_size))
    if n_windows < 3:
        return np.nan
    rri_cumsum = np.cumsum(rri)
    sdnn_ = []
    for i in range(n_windows):
        start = i * window_size
        start_idx = np.where(rri_cumsum >= start)[0][0]
        end_idx = np.where(rri_cumsum < start + window_size)[0][-1]
        sdnn_.append(np.nanstd(rri[start_idx:end_idx], ddof=1))
    sdnni = np.nanmean(sdnn_)
    return sdnni

def _hrv_TINN(rri, bar_x, bar_y, binsize):
    # set pre-defined conditions
    min_error = 2 ** 14
    X = bar_x[np.argmax(bar_y)]  # bin where Y is max
    Y = np.max(bar_y)  # max value of Y
    n = bar_x[np.where(bar_x - np.min(rri) > 0)[0][0]]  # starting search of N
    m = X + binsize  # starting search value of M
    N = 0
    M = 0
    # start to find best values of M and N where least square is minimized
    while n < X:
        while m < np.max(rri):
            n_start = np.where(bar_x == n)[0][0]
            n_end = np.where(bar_x == X)[0][0]
            qn = np.polyval(np.polyfit([n, X], [0, Y], deg=1), bar_x[n_start:n_end + 1])
            m_start = np.where(bar_x == X)[0][0]
            m_end = np.where(bar_x == m)[0][0]
            qm = np.polyval(np.polyfit([X, m], [Y, 0], deg=1), bar_x[m_start:m_end + 1])
            q = np.zeros(len(bar_x))
            q[n_start:n_end + 1] = qn
            q[m_start:m_end + 1] = qm
            # least squares error
            error = np.sum((bar_y[n_start:m_end + 1] - q[n_start:m_end + 1]) ** 2)
            if error < min_error:
                N = n
                M = m
                min_error = error
            m += binsize
        n += binsize
    return M - N

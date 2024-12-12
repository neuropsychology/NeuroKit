# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from ..stats import mad, summary_plot
from .hrv_utils import _hrv_format_input
from .intervals_utils import _intervals_successive


def hrv_time(peaks, sampling_rate=1000, show=False, **kwargs):
    """**Computes time-domain indices of Heart Rate Variability (HRV)**

    Time-domain measures reflect the total variability of HR and are relatively indiscriminate when
    it comes to precisely quantifying the respective contributions of different underlying
    regulatory mechanisms. However, this "general" sensitivity can be seen as a positive feature
    (e.g., in exploratory studies or when specific underlying neurophysiological mechanisms are not
    the focus). Moreover, as they are easy to compute and interpret, time-domain measures are still
    among the most commonly reported HRV indices.

    The time-domain indices can be categorized into deviation-based and difference-based indices
    where the formal are calculated directly from the normal beat-to-beat intervals (normal RR
    intervals or NN intervals), and the later are derived from the difference between successive NN
    intervals.

    .. tip::
        We strongly recommend checking our open-access paper `Pham et al. (2021)
        <https://doi.org/10.3390/s21123998>`_ on HRV indices for more information.

    Parameters
    ----------
    peaks : dict
        Samples at which cardiac extrema (i.e., R-peaks, systolic peaks) occur.
        Can be a list of indices or the output(s) of other functions such as :func:`.ecg_peaks`,
        :func:`.ppg_peaks`, :func:`.ecg_process` or :func:`.bio_process`.
        Can also be a dict containing the keys `RRI` and `RRI_Time`
        to directly pass the R-R intervals and their timestamps, respectively.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks occur. Should be at
        least twice as high as the highest frequency in vhf. By default 1000.
    show : bool
        If ``True``, will plot the distribution of R-R intervals.

    Returns
    -------
    DataFrame
        Contains time domain HRV metrics:

        * **MeanNN**: The mean of the RR intervals.
        * **SDNN**: The standard deviation of the RR intervals.
        * **SDANN1**, **SDANN2**, **SDANN5**: The standard deviation of average RR intervals
          extracted from n-minute segments of time series data (1, 2 and 5 by default). Note that
          these indices require a minimal duration of signal to be computed (3, 6 and 15 minutes
          respectively) and will be silently skipped if the data provided is too short.
        * **SDNNI1**, **SDNNI2**, **SDNNI5**: The mean of the standard deviations of RR intervals
          extracted from n-minute segments of time series data (1, 2 and 5 by default). Note that
          these indices require a minimal duration of signal to be computed (3, 6 and 15 minutes
          respectively) and will be silently skipped if the data provided is too short.
        * **RMSSD**: The square root of the mean of the squared successive differences between
          adjacent RR intervals. It is equivalent (although on another scale) to SD1, and
          therefore it is redundant to report correlations with both (Ciccone, 2017).
        * **SDSD**: The standard deviation of the successive differences between RR intervals.
        * **CVNN**: The standard deviation of the RR intervals (**SDNN**) divided by the mean of
          the RR intervals (**MeanNN**).
        * **CVSD**: The root mean square of successive differences (**RMSSD**) divided by
          the mean of the RR intervals (**MeanNN**).
        * **MedianNN**: The median of the RR intervals.
        * **MadNN**: The median absolute deviation of the RR intervals.
        * **MCVNN**: The median absolute deviation of the RR intervals (**MadNN**) divided by the
          median of the RR intervals (**MedianNN**).
        * **IQRNN**: The interquartile range (**IQR**) of the RR intervals.
        * **SDRMSSD**: SDNN / RMSSD, a time-domain equivalent for the low Frequency-to-High
          Frequency (LF/HF) Ratio (Sollers et al., 2007).
        * **Prc20NN**: The 20th percentile of the RR intervals (Han, 2017; Hovsepian, 2015).
        * **Prc80NN**: The 80th percentile of the RR intervals (Han, 2017; Hovsepian, 2015).
        * **pNN50**: The percentage of absolute differences in successive RR intervals greater than
          50 ms (Bigger et al., 1988; Mietus et al., 2002).
        * **pNN20**: The percentage of absolute differences in successive RR intervals greater than
          20 ms (Mietus et al., 2002).
        * **MinNN**: The minimum of the RR intervals (Parent, 2019; Subramaniam, 2022).
        * **MaxNN**: The maximum of the RR intervals (Parent, 2019; Subramaniam, 2022).
        * **TINN**: A geometrical parameter of the HRV, or more specifically, the baseline width of
          the RR intervals distribution obtained by triangular interpolation, where the error of
          least squares determines the triangle. It is an approximation of the RR interval
          distribution.
        * **HTI**: The HRV triangular index, measuring the total number of RR intervals divided by
          the height of the RR intervals histogram.

    See Also
    --------
    ecg_peaks, ppg_peaks, hrv_frequency, hrv_summary, hrv_nonlinear

    Notes
    -----

    Where applicable, the unit used for these metrics is the millisecond.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Download data
      data = nk.data("bio_resting_5min_100hz")

      # Find peaks
      peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)

      # Compute HRV indices
      @savefig p_hrv_time.png scale=100%
      hrv = nk.hrv_time(peaks, sampling_rate=100, show=True)
      @suppress
      plt.close()

    References
    ----------
    * Bigger Jr, J. T., Kleiger, R. E., Fleiss, J. L., Rolnitzky, L. M., Steinman, R. C., & Miller,
      J. P. (1988). Components of heart rate variability measured during healing of acute myocardial
      infarction. The American journal of cardiology, 61(4), 208-215.
    * Pham, T., Lau, Z. J., Chen, S. H. A., & Makowski, D. (2021). Heart Rate Variability in
      Psychology: A Review of HRV Indices and an Analysis Tutorial. Sensors, 21(12), 3998.
      https://doi.org/10.3390/s21123998
    * Ciccone, A. B., Siedlik, J. A., Wecht, J. M., Deckert, J. A., Nguyen, N. D., & Weir, J. P.
      (2017). Reminder: RMSSD and SD1 are identical heart rate variability metrics. Muscle & nerve,
      56(4), 674-678.
    * Han, L., Zhang, Q., Chen, X., Zhan, Q., Yang, T., & Zhao, Z. (2017). Detecting work-related
      stress with a wearable device. Computers in Industry, 90, 42-49.
    * Hovsepian, K., Al'Absi, M., Ertin, E., Kamarck, T., Nakajima, M., & Kumar, S. (2015). cStress:
      towards a gold standard for continuous stress assessment in the mobile environment. In
      Proceedings of the 2015 ACM international joint conference on pervasive and ubiquitous
      computing (pp. 493-504).
    * Mietus, J. E., Peng, C. K., Henry, I., Goldsmith, R. L., & Goldberger, A. L. (2002). The pNNx
      files: re-examining a widely used heart rate variability measure. Heart, 88(4), 378-380.
    * Parent, M., Tiwari, A., Albuquerque, I., Gagnon, J. F., Lafond, D., Tremblay, S., & Falk, T.
      H. (2019). A multimodal approach to improve the robustness of physiological stress prediction
      during physical activity. In 2019 IEEE International Conference on Systems, Man and
      Cybernetics (SMC) (pp. 4131-4136). IEEE.
    * Stein, P. K. (2002). Assessing heart rate variability from real-world Holter reports. Cardiac
      electrophysiology review, 6(3), 239-244.
    * Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and
      norms. Frontiers in public health, 5, 258.
    * Subramaniam, S. D., & Dass, B. (2022). An Efficient Convolutional Neural Network for Acute
      Pain Recognition Using HRV Features. In Proceedings of the International e-Conference on
      Intelligent Systems and Signal Processing (pp. 119-132). Springer, Singapore.
    * Sollers, J. J., Buchanan, T. W., Mowrer, S. M., Hill, L. K., & Thayer, J. F. (2007).
      Comparison of the ratio of the standard deviation of the RR interval and the root mean
      squared successive differences (SD/rMSSD) to the low frequency-to-high frequency (LF/HF)
      ratio in a patient population and normal healthy controls. Biomed Sci Instrum, 43, 158-163.

    """
    # Sanitize input
    # If given peaks, compute R-R intervals (also referred to as NN) in milliseconds
    rri, rri_time, rri_missing = _hrv_format_input(peaks, sampling_rate=sampling_rate)

    diff_rri = np.diff(rri)

    if rri_missing:
        # Only include successive differences
        diff_rri = diff_rri[_intervals_successive(rri, intervals_time=rri_time)]

    out = {}  # Initialize empty container for results

    # Deviation-based
    out["MeanNN"] = np.nanmean(rri)
    out["SDNN"] = np.nanstd(rri, ddof=1)
    for i in [1, 2, 5]:
        out["SDANN" + str(i)] = _sdann(rri, window=i)
        out["SDNNI" + str(i)] = _sdnni(rri, window=i)

    # Difference-based
    out["RMSSD"] = np.sqrt(np.nanmean(diff_rri**2))
    out["SDSD"] = np.nanstd(diff_rri, ddof=1)

    # Normalized
    out["CVNN"] = out["SDNN"] / out["MeanNN"]
    out["CVSD"] = out["RMSSD"] / out["MeanNN"]

    # Robust
    out["MedianNN"] = np.nanmedian(rri)
    out["MadNN"] = mad(rri)
    out["MCVNN"] = out["MadNN"] / out["MedianNN"]  # Normalized
    out["IQRNN"] = scipy.stats.iqr(rri)
    out["SDRMSSD"] = out["SDNN"] / out["RMSSD"]  # Sollers (2007)
    out["Prc20NN"] = np.nanpercentile(rri, q=20)
    out["Prc80NN"] = np.nanpercentile(rri, q=80)

    # Extreme-based
    nn50 = np.sum(np.abs(diff_rri) > 50)
    nn20 = np.sum(np.abs(diff_rri) > 20)
    out["pNN50"] = nn50 / (len(diff_rri) + 1) * 100
    out["pNN20"] = nn20 / (len(diff_rri) + 1) * 100
    out["MinNN"] = np.nanmin(rri)
    out["MaxNN"] = np.nanmax(rri)

    # Geometrical domain
    binsize = kwargs.get("binsize", ((1 / 128) * 1000))

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


def _sdann(rri, rri_time=None, window=1):
    window_size = window * 60 * 1000  # Convert window in min to ms
    if rri_time is None:
        # Compute the timestamps of the R-R intervals in seconds
        rri_time = np.nancumsum(rri / 1000)
    # Convert timestamps to milliseconds and ensure first timestamp is equal to first interval
    rri_cumsum = (rri_time - rri_time[0]) * 1000 + rri[0]
    n_windows = int(np.round(rri_cumsum[-1] / window_size))
    if n_windows < 3:
        return np.nan
    avg_rri = []
    for i in range(n_windows):
        start = i * window_size
        start_idx = np.where(rri_cumsum >= start)[0][0]
        end_idx = np.where(rri_cumsum < start + window_size)[0][-1]
        avg_rri.append(np.nanmean(rri[start_idx:end_idx]))
    sdann = np.nanstd(avg_rri, ddof=1)
    return sdann


def _sdnni(rri, rri_time=None, window=1):
    window_size = window * 60 * 1000  # Convert window in min to ms
    if rri_time is None:
        # Compute the timestamps of the R-R intervals in seconds
        rri_time = np.nancumsum(rri / 1000)
    # Convert timestamps to milliseconds and ensure first timestamp is equal to first interval
    rri_cumsum = (rri_time - rri_time[0]) * 1000 + rri[0]
    n_windows = int(np.round(rri_cumsum[-1] / window_size))
    if n_windows < 3:
        return np.nan
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
    min_error = 2**14
    X = bar_x[np.argmax(bar_y)]  # bin where Y is max
    Y = np.max(bar_y)  # max value of Y
    idx_where = np.where(bar_x - np.min(rri) > 0)[0]
    if len(idx_where) == 0:
        return np.nan
    n = bar_x[idx_where[0]]  # starting search of N
    m = X + binsize  # starting search value of M
    N = 0
    M = 0
    # start to find best values of M and N where least square is minimized
    while n < X:
        while m < np.max(rri):
            n_start = np.where(bar_x == n)[0][0]
            n_end = np.where(bar_x == X)[0][0]
            qn = np.polyval(
                np.polyfit([n, X], [0, Y], deg=1), bar_x[n_start : n_end + 1]
            )
            m_start = np.where(bar_x == X)[0][0]
            m_end = np.where(bar_x == m)[0][0]
            qm = np.polyval(
                np.polyfit([X, m], [Y, 0], deg=1), bar_x[m_start : m_end + 1]
            )
            q = np.zeros(len(bar_x))
            q[n_start : n_end + 1] = qn
            q[m_start : m_end + 1] = qm
            # least squares error
            error = np.sum((bar_y[n_start : m_end + 1] - q[n_start : m_end + 1]) ** 2)
            if error < min_error:
                N = n
                M = m
                min_error = error
            m += binsize
        n += binsize
    return M - N

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

from . import hrv_time, hrv_frequency, hrv_nonlinear
from ..signal import signal_rate
from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize


def hrv_summary(heart_period, peaks=None, sampling_rate=1000, show=False):
    """ Computes indices of Heart Rate Variability (HRV).

    Note that a minimum recording is recommended for somenindices to be
    meaninful. For instance, 1, 2 and 5 minutes of good signal are the
    recomended minimums for HF, LF and LF/HF, respectively.

    Parameters
    ----------
    heart_period : array
        Array containing the heart period as returned by `signal_period()`.
    peaks : dict
        The samples at which the peaks occur. Returned by `ecg_peaks()` or
        `ppg_peaks`. Defaults to None.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    show : bool
        If True, will return a Poincaré plot, a scattergram, which plots each
        RR interval against the next successive one. The ellipse centers around
        the average RR interval. Defaults to False.

    Returns
    -------
    DataFrame
        DataFrame consisting of the computed HRV metrics, which includes:
            - **Time Domain**
                - "*HRV_RMSSD*": the square root of the mean of the sum of successive differences between adjacent RR intervals.
                - "*HRV_MeanNN*": the mean of the RR intervals.
                - "*HRV_SDNN*": the standard deviation of the RR intervals.
                - "*HRV_SDSD*": the standard deviation of the successive differences between RR intervals.
                - "*HRV_CVNN*": the standard deviation of the RR intervals (SDNN) divided by the mean of the RR intervals (MeanNN).
                - "*HRV_CVSD*": the root mean square of the sum of successive differences (RMSSD) divided by the mean of the RR intervals (MeanNN).
                - "*HRV_MedianNN*": the median of the absolute values of the successive differences between RR intervals.
                - "*HRV_MadNN*": the median absolute deviation of the RR intervals.
                - "*HCVNN*": the median absolute deviation of the RR intervals (MadNN) divided by the median of the absolute differences of their successive differences (MedianNN).
                - "*pNN50*": the proportion of RR intervals greater than 50ms, out of the total number of RR intervals.
                - "*pNN20*": the proportion of RR intervals greater than 20ms, out of the total number of RR intervals.
                - "*HRV_TINN*": a geometrical parameter of the HRV, or more specifically, the baseline width of the RR intervals distribution obtained by triangular interpolation, where the error of least squares determines the triangle. It is an approximation of the RR interval distribution.
                - "*HRV_HTI*": the HRV triangular index, measuring the total number of RR intervals divded by the height of the RR intervals histogram.

            - **Frequency Domain**:
                - "*HRV_ULF*": spectral power density pertaining to ultra low frequency band i.e., .0 to .0033 Hz by default.
                - "*HRV_VLF*": spectral power density pertaining to very low frequency band i.e., .0033 to .04 Hz by default.
                - "*HRV_LF*": spectral power density pertaining to low frequency band i.e., .04 to .15 Hz by default.
                - "*HRV_HF*": spectral power density pertaining to high frequency band i.e., .15 to .4 Hz by default.
                - "*HRV_VHF*": variability, or signal power, in very high frequency i.e., .4 to .5 Hz by default.
                - "*HRV_LFHF*": the ratio of low frequency power to high frequency power.
                - "*HRV_LFn*": the normalized low frequency, obtained by dividing the low frequency power by the total power.
                - "*HRV_HFn*": the normalized high frequency, obtained by dividing the low frequency power by the total power.

            - **Non-linear Domain**:
                - "*HRV_SD1*": SD1 is a measure of the spread of RR intervals on the Poincaré plot perpendicular to the line of identity. It is an index of short-term RR interval fluctuations i.e., beat-to-beat variability.
                - "*HRV_SD2*": SD2 is a measure of the spread of RR intervals on the Poincaré plot along the line of identity. It is an index of long-term RR interval fluctuations.
                - "*HRV_SD2SD1*": the ratio between short and long term fluctuations of the RR intervals (SD2 divided by SD1).
                - "*HRV_CSI*": the Cardiac Sympathetic Index, calculated by dividing the longitudinal variability of the Poincaré plot by its transverse variability.
                - "*HRV_CVI*": the Cardiac Vagal Index, equal to the logarithm of the product of longitudinal and transverse variability.
                - "*HRV_CSI_Modified*": the modified CSI obtained by dividing the square of the longitudinal variability by its transverse variability. Usually used in seizure research.
                - "*HRV_SampEn*": the sample entropy measure of HRV, calculated by `entropy_sample()`.

    See Also
    --------
    ecg_peak, signal_power, signal_rate, entropy_sample

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=240, sampling_rate=1000)
    >>> ecg, info = nk.ecg_process(ecg, sampling_rate=1000)
    >>> hrv = nk.ecg_hrv(ecg, sampling_rate=1000, show=True)
    >>> hrv
    >>> hrv[["HRV_HF"]]
    >>>
    >>> ecg = nk.ecg_simulate(duration=240, sampling_rate=200)
    >>> ecg, info = nk.ecg_process(ecg, sampling_rate=200)
    >>> hrv = nk.ecg_hrv(ecg, sampling_rate=200)

    References
    ----------
    - Stein, P. K. (2002). Assessing heart rate variability from real-world
      Holter reports. Cardiac electrophysiology review, 6(3), 239-244.
    - Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. Frontiers in public health, 5, 258.
    """
    # Sanitize input
    heart_period = heart_period[peaks] * 1000  # milliseconds
    heart_period_intp = 60 / heart_period * 1000  # milliseconds

    # Get indices
    hrv = {}  # Initialize empty dict
    hrv.update(hrv_time(heart_period))
    hrv.update(hrv_frequency(heart_period_intp, sampling_rate, show=show))
    hrv.update(hrv_nonlinear(heart_period, heart_period_intp))

    hrv = pd.DataFrame.from_dict(hrv, orient='index').T.add_prefix("HRV_")

    if show:
        _hrv_plot(heart_period, heart_period_intp)

    return hrv


def _hrv_formatinput(heart_rate, peaks=None, sampling_rate=1000):

    if isinstance(heart_rate, tuple):
        heart_rate = heart_rate[0]
        peaks = None

    if isinstance(heart_rate, pd.DataFrame):
        df = heart_rate.copy()
        cols = [col for col in df.columns if 'heart_rate' in col]
        if len(cols) == 0:
            cols = [col for col in df.columns if 'ECG_R_Peaks' in col]
            if len(cols) == 0:
                raise ValueError("NeuroKit error: _hrv_formatinput(): Wrong input, ",
                                 "we couldn't extract heart_rate and peaks indices.")
            else:
                heart_rate = signal_rate(peaks, sampling_rate=sampling_rate,
                                       desired_length=len(df))
        else:
            heart_rate = df[cols[0]].values

    if peaks is None:
        try:
            peaks, _ = _signal_formatpeaks_sanitize(df, desired_length=None)
        except NameError:
            raise ValueError("NeuroKit error: _hrv_formatinput(): Wrong input, ",
                             "we couldn't extract peaks indices.")
    else:
        peaks, _ = _signal_formatpeaks_sanitize(peaks, desired_length=None)

    return heart_rate, peaks


def _hrv_plot(heart_period, heart_period_intp):
    # Axes
    ax1 = heart_period[:-1]
    ax2 = heart_period[1:]

    # Compute features
    poincare_features = hrv_nonlinear(heart_period, heart_period_intp)
    sd1 = poincare_features["SD1"]
    sd2 = poincare_features["SD2"]
    mean_heart_period = np.mean(heart_period)

    # Plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plt.title("Poincaré Plot", fontsize=20)
    plt.xlabel('RR_n (s)', fontsize=15)
    plt.ylabel('RR_n+1 (s)', fontsize=15)
    plt.xlim(min(heart_period) - 10, max(heart_period) + 10)
    plt.ylim(min(heart_period) - 10, max(heart_period) + 10)
    ax.scatter(ax1, ax2, c='b', s=4)

    # Ellipse plot feature
    ellipse = matplotlib.patches.Ellipse(xy=(mean_heart_period,
                                             mean_heart_period),
                                         width=2 * sd2 + 1, height=2 * sd1 + 1,
                                         angle=45, linewidth=2, fill=False)
    ax.add_patch(ellipse)
    ellipse = matplotlib.patches.Ellipse(xy=(mean_heart_period,
                                             mean_heart_period), width=2 * sd2,
                                         height=2 * sd1, angle=45)
    ellipse.set_alpha(0.02)
    ellipse.set_facecolor("blue")
    ax.add_patch(ellipse)

    # Arrow plot feature
    sd1_arrow = ax.arrow(mean_heart_period, mean_heart_period,
                         -sd1 * np.sqrt(2) / 2, sd1 * np.sqrt(2) / 2,
                         linewidth=3, ec='r', fc="r", label="SD1")
    sd2_arrow = ax.arrow(mean_heart_period,
                         mean_heart_period, sd2 * np.sqrt(2) / 2,
                         sd2 * np.sqrt(2) / 2,
                         linewidth=3, ec='y', fc="y", label="SD2")

    plt.legend(handles=[sd1_arrow, sd2_arrow], fontsize=12, loc="best")

    return fig

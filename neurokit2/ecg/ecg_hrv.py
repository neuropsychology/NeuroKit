import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

from .ecg_rate import ecg_rate as nk_ecg_rate
from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize
from ..signal import signal_power
from ..stats import mad
from ..complexity import entropy_sample


def ecg_hrv(ecg_rate, rpeaks=None, sampling_rate=1000, show=False):
    """ Computes indices of Heart Rate Variability (HRV).

    Different metrics of HRV exist. Note that a minimum recording is recommended for some
    indices to be meaninful. For instance, 1, 2 and 5 minutes of good signal are the
    recomended minimums for HF, LF and LF/HF, respectively.

    Parameters
    ----------
    ecg_rate : array
        Array containing the heart rate, produced by `ecg_rate()`.
    rpeaks : dict
        The samples at which the R-peaks occur. Dict returned by
        `ecg_peaks()`. Defaults to None.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    show : bool
        If True, will return a Poincaré plot, a scattergram, which plots each RR interval against the next successive one. The ellipse centers around the average RR interval. Defaults to False.

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
    ecg_rate, ecg_peak, signal_power, entropy_sample

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
    ecg_rate, rpeaks = _ecg_hrv_formatinput(ecg_rate, rpeaks, sampling_rate)

    # Get raw and interpolated R-R intervals
    rri = np.diff(rpeaks) / sampling_rate * 1000
    ecg_period = ecg_rate / 60 * 1000  # Express in milliseconds

    # Get indices
    hrv = {}  # Initialize empty dict
    hrv.update(_ecg_hrv_time(rri))
    hrv.update(_ecg_hrv_frequency(ecg_period, sampling_rate, show=show))
    hrv.update(_ecg_hrv_nonlinear(rri, ecg_period))

    hrv = pd.DataFrame.from_dict(hrv, orient='index').T.add_prefix("HRV_")

    if show:
        _ecg_hrv_plot(rri, ecg_period)

    return hrv


# =============================================================================
# Methods (Domains)
# =============================================================================



def _ecg_hrv_time(rri):
    diff_rri = np.diff(rri)
    out = {}  # Initialize empty dict

    # Mean based
    out["RMSSD"] = np.sqrt(np.mean(diff_rri ** 2))
    out["MeanNN"] = np.mean(rri)
    out["SDNN"] = np.std(rri, ddof=1)
    out["SDSD"] = np.std(diff_rri, ddof=1)
    out["CVNN"] = out["SDNN"] / out["MeanNN"]
    out["CVSD"] = out["RMSSD"] / out["MeanNN"]

    # Robust
    out["MedianNN"] = np.median(np.abs(rri))
    out["MadNN"] = mad(rri)
    out["MCVNN"] = out["MadNN"] / out["MedianNN"]

    # Extreme-based
    nn50 = np.sum(np.abs(diff_rri) > 50)
    nn20 = np.sum(np.abs(diff_rri) > 20)
    out["pNN50"] = nn50 / len(rri) * 100
    out["pNN20"] = nn20 / len(rri) * 100

    # Geometrical domain
    bar_y, bar_x = np.histogram(rri, bins=range(300, 2000, 8))
    bar_y, bar_x = np.histogram(rri, bins="auto")
    out["TINN"] = np.max(bar_x) - np.min(bar_x)  # Triangular Interpolation of the NN Interval Histogram
    out["HTI"] = len(rri) / np.max(bar_y)  # HRV Triangular Index

    return out





def _ecg_hrv_frequency(ecg_period, sampling_rate=1000, ulf=(0, 0.0033), vlf=(0.0033, 0.04), lf=(0.04, 0.15), hf=(0.15, 0.4), vhf=(0.4, 0.5), method="welch", show=False):
    power = signal_power(ecg_period, frequency_band=[ulf, vlf, lf, hf, vhf], sampling_rate=sampling_rate, method=method, max_frequency=0.5, show=show)
    power.columns = ["ULF", "VLF", "LF", "HF", "VHF"]
    out = power.to_dict(orient="index")[0]

    # Normalized
    total_power = np.sum(power.values)
    out["LFHF"] = out["LF"] / out["HF"]
    out["LFn"] = out["LF"] / total_power
    out["HFn"] = out["HF"] / total_power

    # Log
    out["LnHF"] = np.log(out["HF"])
    return out





def _ecg_hrv_nonlinear(rri, ecg_period):
    diff_rri = np.diff(rri)
    out = {}

    # Poincaré plot
    out["SD1"] = np.sqrt(np.std(diff_rri, ddof=1) ** 2 * 0.5)
    out["SD2"] = np.sqrt(2 * np.std(rri, ddof=1) ** 2 - 0.5 * np.std(diff_rri, ddof=1) ** 2)
    out["SD2SD1"] = out["SD2"] / out["SD1"]

    # CSI / CVI
    T = 4 * out["SD1"]
    L = 4 * out["SD2"]
    out["CSI"] = L / T
    out["CVI"] = np.log10(L * T)
    out["CSI_Modified"] = L ** 2 / T

    # Entropy
    out["SampEn"] = entropy_sample(rri, order=2, r=0.2*np.std(rri, ddof=1))
    return out


# =============================================================================
# Internals
# =============================================================================

def _ecg_hrv_formatinput(ecg_rate, rpeaks=None, sampling_rate=1000):

    if isinstance(ecg_rate, tuple):
        ecg_rate = ecg_rate[0]
        rpeaks = None

    if isinstance(ecg_rate, pd.DataFrame):
        df = ecg_rate.copy()
        cols = [col for col in df.columns if 'ECG_Rate' in col]
        if len(cols) == 0:
            cols = [col for col in df.columns if 'ECG_R_Peaks' in col]
            if len(cols) == 0:
                raise ValueError("NeuroKit error: _ecg_hrv_formatinput(): Wrong input, ",
                                 "we couldn't extract ecg_rate and rpeaks indices.")
            else:
                ecg_rate = nk_ecg_rate(rpeaks, sampling_rate=sampling_rate, desired_length=len(df))
        else:
            ecg_rate = df[cols[0]].values



    if rpeaks is None:
        try:
            rpeaks, _ = _signal_formatpeaks_sanitize(df, desired_length=None)
        except NameError:
            raise ValueError("NeuroKit error: _ecg_hrv_formatinput(): Wrong input, ",
                             "we couldn't extract rpeaks indices.")
    else:
        rpeaks, _ = _signal_formatpeaks_sanitize(rpeaks, desired_length=None)

    return ecg_rate, rpeaks



def _ecg_hrv_plot(rri, ecg_period):
    # Axes
    ax1 = rri[:-1]
    ax2 = rri[1:]

    # Compute features
    poincare_features = _ecg_hrv_nonlinear(rri, ecg_period)
    sd1 = poincare_features["SD1"]
    sd2 = poincare_features["SD2"]
    mean_rri = np.mean(rri)

    # Plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plt.title("Poincaré Plot", fontsize=20)
    plt.xlabel('RR_n (s)', fontsize=15)
    plt.ylabel('RR_n+1 (s)', fontsize=15)
    plt.xlim(min(rri) - 10, max(rri) + 10)
    plt.ylim(min(rri) - 10, max(rri) + 10)
    ax.scatter(ax1, ax2, c='b', s=4)

    # Ellipse plot feature
    ellipse = matplotlib.patches.Ellipse(xy=(mean_rri, mean_rri), width=2 * sd2 + 1,
                                         height=2 * sd1 + 1, angle=45, linewidth=2,
                                         fill=False)
    ax.add_patch(ellipse)
    ellipse = matplotlib.patches.Ellipse(xy=(mean_rri, mean_rri), width=2 * sd2,
                                         height=2 * sd1, angle=45)
    ellipse.set_alpha(0.02)
    ellipse.set_facecolor("blue")
    ax.add_patch(ellipse)

    # Arrow plot feature
    sd1_arrow = ax.arrow(mean_rri, mean_rri, -sd1 * np.sqrt(2) / 2, sd1 * np.sqrt(2) / 2,
                         linewidth=3, ec='r', fc="r", label="SD1")
    sd2_arrow = ax.arrow(mean_rri, mean_rri, sd2 * np.sqrt(2) / 2, sd2 * np.sqrt(2) / 2,
                         linewidth=3, ec='y', fc="y", label="SD2")

    plt.legend(handles=[sd1_arrow, sd2_arrow], fontsize=12, loc="best")

    return fig

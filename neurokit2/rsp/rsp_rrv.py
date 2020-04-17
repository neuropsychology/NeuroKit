# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

from .rsp_rate import rsp_rate as nk_rsp_rate
from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize
from ..signal import signal_power
from ..complexity import entropy_sample
from ..complexity import entropy_approximate
from ..complexity import complexity_dfa


def rsp_rrv(rsp_rate, peaks=None, sampling_rate=1000, show=False):
    """Computes time domain and frequency domain features for Respiratory Rate Variability (RRV) analysis.

    Parameters
    ----------
    rsp_rate : array
        Array containing the respiratory rate, produced by `rsp_rate()`.
    peaks : dict
        The samples at which the inhalation peaks occur.
        Dict returned by `rsp_peaks()`. Defaults to None.
    sampling_rate : int
        The sampling frequency of the signal
        (in Hz, i.e., samples/second).
    show : bool
        If True, will return a Poincaré plot, a scattergram, which plots each breath-to-breath interval against the next successive one. The ellipse centers around the average breath-to-breath interval. Defaults to False.

    Returns
    -------
    DataFrame
        DataFrame consisting of the computed RRV metrics, which includes:
            - "*RRV_SDBB*": the standard deviation of the breath-to-breath intervals.
            - "*RRV_RMSSD*": the root mean square of successive differences of the breath-to-breath intervals.
            - "*RRV_SDSD*": the standard deviation of the successive differences between adjacent breath-to-breath intervals.
            - "*RRV_BBx*": the number of successive interval differences that are greater than x seconds.
            - "*RRV-pBBx*": the proportion of breath-to-breath intervals that are greater than x seconds, out of the total number of intervals.
            - "*RRV_VLF*": spectral power density pertaining to very low frequency band i.e., 0 to .04 Hz by default.
            - "*RRV_LF*": spectral power density pertaining to low frequency band i.e., .04 to .15 Hz by default.
            - "*RRV_HF*": spectral power density pertaining to high frequency band i.e., .15 to .4 Hz by default.
            - "*RRV_LFHF*": the ratio of low frequency power to high frequency power.
            - "*RRV_LFn*": the normalized low frequency, obtained by dividing the low frequency power by the total power.
            - "*RRV_HFn*": the normalized high frequency, obtained by dividing the low frequency power by total power.
            - "*RRV_SD1*": SD1 is a measure of the spread of breath-to-breath intervals on the Poincaré plot perpendicular to the line of identity. It is an index of short-term variability.
            - "*RRV_SD2*": SD2 is a measure of the spread of breath-to-breath intervals on the Poincaré plot along the line of identity. It is an index of long-term variability.
            - "*RRV_SD2SD1*": the ratio between short and long term fluctuations of the breath-to-breath intervals (SD2 divided by SD1).
            - "*RRV_ApEn*": the approximate entropy of RRV, calculated by `entropy_approximate()`.
            - "*RRV_SampEn*": the sample entropy of RRV, calculated by `entropy_sample()`.
            - "*RRV_DFA*": the fluctuation value generated from Detrended Fluctuation Analysis i.e. the root mean square deviation from the fitted trend of the breath-to-breath intervals. However, this is designed to analyze time series data over several hours.

    See Also
    --------
    rsp_rate, rsp_peaks, signal_power, entropy_sample, entropy_approximate

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration=90, respiratory_rate=15)
    >>> rsp, info = nk.rsp_process(rsp)
    >>> rrv = nk.rsp_rrv(rsp, show=True)

    References
    ----------
    - Soni, R., & Muniyandi, M. (2019). Breath rate variability:
    a novel measure to study the meditation effects. International Journal of Yoga,
    12(1), 45.
    """
    # Sanitize input
    rsp_rate, peaks = _rsp_rrv_formatinput(rsp_rate, peaks, sampling_rate)

    # Get raw and interpolated R-R intervals
    bbi = np.diff(peaks) / sampling_rate * 1000
    rsp_period = 60 * sampling_rate/ rsp_rate

    # Get indices
    rrv = {}  # Initialize empty dict
    rrv.update(_rsp_rrv_time(bbi))
    rrv.update(_rsp_rrv_frequency(rsp_period, show=show))
    rrv.update(_rsp_rrv_nonlinear(bbi, rsp_period))

    rrv = pd.DataFrame.from_dict(rrv, orient='index').T.add_prefix("RRV_")

    if show:
        _rsp_rrv_plot(bbi, rsp_period)

    return rrv


# =============================================================================
# Methods (Domains)
# =============================================================================


def _rsp_rrv_time(bbi):
    diff_bbi = np.diff(bbi)
    out = {}  # Initialize empty dict

    # Mean based
    out["SDBB"] = np.std(bbi, ddof=1)
    out["RMSSD"] = np.sqrt(np.mean(diff_bbi ** 2))
    out["SDSD"] = np.std(diff_bbi, ddof=1)
#    out["MeanNN"] = np.mean(rri)
#    out["CVNN"] = out["SDNN"] / out["MeanNN"]
#    out["CVSD"] = out["RMSSD"] / out["MeanNN"]

    # Robust
#    out["MedianNN"] = np.median(np.abs(rri))
#    out["MadNN"] = mad(rri)
#    out["MCVNN"] = out["MadNN"] / out["MedianNN"]

#    # Extreme-based
#    nn50 = np.sum(np.abs(diff_rri) > 50)
#    nn20 = np.sum(np.abs(diff_rri) > 20)
#    out["pNN50"] = nn50 / len(rri) * 100
#    out["pNN20"] = nn20 / len(rri) * 100
#
#    # Geometrical domain
#    bar_y, bar_x = np.histogram(rri, bins=range(300, 2000, 8))
#    bar_y, bar_x = np.histogram(rri, bins="auto")
#    out["TINN"] = np.max(bar_x) - np.min(bar_x)  # Triangular Interpolation of the NN Interval Histogram
#    out["HTI"] = len(rri) / np.max(bar_y)  # HRV Triangular Index

    return out





def _rsp_rrv_frequency(rsp_period, vlf=(0, 0.04), lf=(0.04, 0.15), hf=(0.15, 0.4), method="welch", show=False):
    power = signal_power(rsp_period, frequency_band=[vlf, lf, hf], sampling_rate=1000, method=method, max_frequency=0.5, show=show)
    power.columns = ["VLF", "LF", "HF"]
    out = power.to_dict(orient="index")[0]

    # Normalized
    total_power = np.sum(power.values)
    out["LFHF"] = out["LF"] / out["HF"]
    out["LFn"] = out["LF"] / total_power
    out["HFn"] = out["HF"] / total_power

    return out




def _rsp_rrv_nonlinear(bbi, rsp_period):
    diff_bbi = np.diff(bbi)
    out = {}

    # Poincaré plot
    out["SD1"] = np.sqrt(np.std(diff_bbi, ddof=1) ** 2 * 0.5)
    out["SD2"] = np.sqrt(2 * np.std(bbi, ddof=1) ** 2 - 0.5 * np.std(diff_bbi, ddof=1) ** 2)
    out["SD2SD1"] = out["SD2"] / out["SD1"]

    # CSI / CVI
#    T = 4 * out["SD1"]
#    L = 4 * out["SD2"]
#    out["CSI"] = L / T
#    out["CVI"] = np.log10(L * T)
#    out["CSI_Modified"] = L ** 2 / T

    # Entropy
    out["ApEn"] = entropy_approximate(bbi, order=2)
    out["SampEn"] = entropy_sample(bbi, order=2, r=0.2*np.std(bbi, ddof=1))

    # DFA
    out["DFA"] = complexity_dfa(bbi, order=1)

    return out


# =============================================================================
# Internals
# =============================================================================

def _rsp_rrv_formatinput(rsp_rate, peaks, sampling_rate=1000):

    if isinstance(rsp_rate, tuple):
        rsp_rate = rsp_rate[0]
        peaks = None

    if isinstance(rsp_rate, pd.DataFrame):
        df = rsp_rate.copy()
        cols = [col for col in df.columns if 'RSP_Rate' in col]
        if len(cols) == 0:
            cols = [col for col in df.columns if 'RSP_Peaks' in col]
            if len(cols) == 0:
                raise ValueError("NeuroKit error: _rsp_rrv_formatinput(): Wrong input,"
                                 "we couldn't extract rsp_rate and peaks indices.")
            else:
                rsp_rate = nk_rsp_rate(peaks, sampling_rate=sampling_rate, desired_length=len(df))
        else:
            rsp_rate = df[cols[0]].values

    if peaks is None:
        try:
            peaks, _ = _signal_formatpeaks_sanitize(df, desired_length=None, key="RSP_Peaks")
        except NameError:
            raise ValueError("NeuroKit error: _rsp_rrv_formatinput():"
                             "Wrong input, we couldn't extract"
                             "respiratory peaks indices.")
    else:
        peaks, _ = _signal_formatpeaks_sanitize(peaks, desired_length=None, key="RSP_Peaks")

    return rsp_rate, peaks



def _rsp_rrv_plot(bbi, rsp_period):
    # Axes
    ax1 = bbi[:-1]
    ax2 = bbi[1:]

    # Compute features
    poincare_features = _rsp_rrv_nonlinear(bbi, rsp_period)
    sd1 = poincare_features["SD1"]
    sd2 = poincare_features["SD2"]
    mean_bbi = np.mean(bbi)

    # Plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plt.title("Poincaré Plot", fontsize=20)
    plt.xlabel('BB_n (s)', fontsize=15)
    plt.ylabel('BB_n+1 (s)', fontsize=15)
    plt.xlim(min(bbi) - 10, max(bbi) + 10)
    plt.ylim(min(bbi) - 10, max(bbi) + 10)
    ax.scatter(ax1, ax2, c='b', s=4)

    # Ellipse plot feature
    ellipse = matplotlib.patches.Ellipse(xy=(mean_bbi, mean_bbi), width=2 * sd2 + 1,
                                         height=2 * sd1 + 1, angle=45, linewidth=2,
                                         fill=False)
    ax.add_patch(ellipse)
    ellipse = matplotlib.patches.Ellipse(xy=(mean_bbi, mean_bbi), width=2 * sd2,
                                         height=2 * sd1, angle=45)
    ellipse.set_alpha(0.02)
    ellipse.set_facecolor("blue")
    ax.add_patch(ellipse)

    # Arrow plot feature
    sd1_arrow = ax.arrow(mean_bbi, mean_bbi, -sd1 * np.sqrt(2) / 2, sd1 * np.sqrt(2) / 2,
                         linewidth=3, ec='r', fc="r", label="SD1")
    sd2_arrow = ax.arrow(mean_bbi, mean_bbi, sd2 * np.sqrt(2) / 2, sd2 * np.sqrt(2) / 2,
                         linewidth=3, ec='y', fc="y", label="SD2")

    plt.legend(handles=[sd1_arrow, sd2_arrow], fontsize=12, loc="best")

    return fig

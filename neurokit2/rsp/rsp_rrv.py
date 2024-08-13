# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..complexity import entropy_approximate, entropy_sample, fractal_dfa
from ..misc import NeuroKitWarning
from ..signal import signal_power, signal_rate
from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize
from ..stats import mad


def rsp_rrv(rsp_rate, troughs=None, sampling_rate=1000, show=False, silent=True):
    """**Respiratory Rate Variability (RRV)**

    Computes time domain and frequency domain features for Respiratory Rate Variability (RRV)
    analysis.

    Parameters
    ----------
    rsp_rate : array
        Array containing the respiratory rate, produced by :func:`.signal_rate`.
    troughs : dict
        The samples at which the inhalation onsets occur.
        Dict returned by :func:`rsp_peaks` (Accessible with the key, ``"RSP_Troughs"``).
        Defaults to ``None``.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    show : bool
        If ``True``, will return a Poincaré plot, a scattergram, which plots each breath-to-breath
        interval against the next successive one. The ellipse centers around the average
        breath-to-breath interval. Defaults to ``False``.
    silent : bool
        If ``False``, warnings will be printed. Default to ``True``.

    Returns
    -------
    DataFrame
        DataFrame consisting of the computed RRV metrics, which includes:

        .. codebookadd::
            RRV_SDBB|The standard deviation of the breath-to-breath intervals.
            RRV_RMSSD|The root mean square of successive differences of the breath-to-breath intervals.
            RRV_SDSD|The standard deviation of the successive differences between adjacent \
                breath-to-breath intervals.
            RRV_BBx|The number of successive interval differences that are greater than x seconds.
            RRV_pBBx|the proportion of breath-to-breath intervals that are greater than x seconds, \
                out of the total number of intervals.
            RRV_VLF|Spectral power density pertaining to very low frequency band (i.e., 0 to\
                .04 Hz) by default.
            RRV_LF|Spectral power density pertaining to low frequency band (i.e., .04 to \
                .15 Hz) by default.
            RRV_HF|Spectral power density pertaining to high frequency band (i.e., .15 to \
                .4 Hz) by default.
            RRV_LFHF|The ratio of low frequency power to high frequency power.
            RRV_LFn|The normalized low frequency, obtained by dividing the low frequency power by \
                the total power.
            RRV_HFn|The normalized high frequency, obtained by dividing the low frequency power by \
                total power.
            RRV_SD1|SD1 is a measure of the spread of breath-to-breath intervals on the Poincaré \
                plot perpendicular to the line of identity. It is an index of short-term variability.
            RRV_SD2|SD2 is a measure of the spread of breath-to-breath intervals on the Poincaré \
                plot along the line of identity. It is an index of long-term variability.
            RRV_SD2SD1|The ratio between short and long term fluctuations of the breath-to-breath \
                intervals (SD2 divided by SD1).
            RRV_DFA_alpha1|The "short-term" fluctuation value generated from Detrended Fluctuation \
                Analysis i.e. the root mean square deviation from the fitted trend of the \
                breath-to-breath intervals. Will only be computed if mora than 160 breath cycles \
                in the signal.
            RRV_DFA_alpha2|The long-term fluctuation value. Will only be computed if mora than \
                640 breath cycles in the signal.
            RRV_ApEn|The approximate entropy of RRV, calculated by :func:`.entropy_approximate`.
            RRV_SampEn|The sample entropy of RRV, calculated by :func:`.entropy_sample`.

        * **MFDFA indices**: Indices related to the :func:`multifractal spectrum <.fractal_dfa()>`.


    See Also
    --------
    signal_rate, rsp_peaks, signal_power, entropy_sample, entropy_approximate

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      rsp = nk.rsp_simulate(duration=90, respiratory_rate=15)
      rsp, info = nk.rsp_process(rsp)
      nk.rsp_rrv(rsp, show=True)


    References
    ----------
    * Soni, R., & Muniyandi, M. (2019). Breath rate variability: a novel measure to study the
      meditation effects. International Journal of Yoga, 12(1), 45.

    """
    # Sanitize input
    rsp_rate, troughs = _rsp_rrv_formatinput(rsp_rate, troughs, sampling_rate)

    # Get raw and interpolated R-R intervals
    bbi = np.diff(troughs) / sampling_rate * 1000
    rsp_period = 60 * sampling_rate / rsp_rate

    # Get indices
    rrv = {}  # Initialize empty dict
    rrv.update(_rsp_rrv_time(bbi))
    rrv.update(
        _rsp_rrv_frequency(rsp_period, sampling_rate=sampling_rate, show=show, silent=silent)
    )
    rrv.update(_rsp_rrv_nonlinear(bbi))

    rrv = pd.DataFrame.from_dict(rrv, orient="index").T.add_prefix("RRV_")

    if show:
        _rsp_rrv_plot(bbi)

    return rrv


# =============================================================================
# Methods (Domains)
# =============================================================================


def _rsp_rrv_time(bbi):
    diff_bbi = np.diff(bbi)
    out = {}  # Initialize empty dict

    # Mean based
    out["RMSSD"] = np.sqrt(np.mean(diff_bbi**2))

    out["MeanBB"] = np.nanmean(bbi)
    out["SDBB"] = np.nanstd(bbi, ddof=1)
    out["SDSD"] = np.nanstd(diff_bbi, ddof=1)

    out["CVBB"] = out["SDBB"] / out["MeanBB"]
    out["CVSD"] = out["RMSSD"] / out["MeanBB"]

    # Robust
    out["MedianBB"] = np.nanmedian(bbi)
    out["MadBB"] = mad(bbi)
    out["MCVBB"] = out["MadBB"] / out["MedianBB"]

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


def _rsp_rrv_frequency(
    rsp_period,
    vlf=(0, 0.04),
    lf=(0.04, 0.15),
    hf=(0.15, 0.4),
    sampling_rate=1000,
    method="welch",
    show=False,
    silent=True,
):
    power = signal_power(
        rsp_period,
        frequency_band=[vlf, lf, hf],
        sampling_rate=sampling_rate,
        method=method,
        max_frequency=0.5,
        show=show,
    )
    power.columns = ["VLF", "LF", "HF"]
    out = power.to_dict(orient="index")[0]

    if silent is False:
        for frequency in out.keys():
            if out[frequency] == 0.0:
                warn(
                    "The duration of recording is too short to allow"
                    " reliable computation of signal power in frequency band " + frequency + "."
                    " Its power is returned as zero.",
                    category=NeuroKitWarning,
                )

    # Normalized
    total_power = np.sum(power.values)
    out["LFHF"] = out["LF"] / out["HF"]
    out["LFn"] = out["LF"] / total_power
    out["HFn"] = out["HF"] / total_power

    return out


def _rsp_rrv_nonlinear(bbi):
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
    out["ApEn"] = entropy_approximate(bbi, dimension=2)[0]
    out["SampEn"] = entropy_sample(bbi, dimension=2, tolerance=0.2 * np.std(bbi, ddof=1))[0]

    # DFA
    if len(bbi) / 10 > 16:
        out["DFA_alpha1"] = fractal_dfa(bbi, scale=np.arange(4, 17), multifractal=False)[0]
        # For multifractal
        mdfa_alpha1, _ = fractal_dfa(
            bbi, multifractal=True, q=np.arange(-5, 6), scale=np.arange(4, 17)
        )
        for k in mdfa_alpha1.columns:
            out["MFDFA_alpha1_" + k] = mdfa_alpha1[k].values[0]

    if len(bbi) > 65:
        out["DFA_alpha2"] = fractal_dfa(bbi, scale=np.arange(16, 65), multifractal=False)[0]
        # For multifractal
        mdfa_alpha2, _ = fractal_dfa(
            bbi, multifractal=True, q=np.arange(-5, 6), scale=np.arange(16, 65)
        )
        for k in mdfa_alpha2.columns:
            out["MFDFA_alpha2_" + k] = mdfa_alpha2[k].values[0]
    return out


# =============================================================================
# Internals
# =============================================================================


def _rsp_rrv_formatinput(rsp_rate, troughs, sampling_rate=1000):

    if isinstance(rsp_rate, tuple):
        rsp_rate = rsp_rate[0]
        troughs = None

    if isinstance(rsp_rate, pd.DataFrame):
        df = rsp_rate.copy()
        cols = [col for col in df.columns if "RSP_Rate" in col]
        if len(cols) == 0:
            cols = [col for col in df.columns if "RSP_Troughs" in col]
            if len(cols) == 0:
                raise ValueError(
                    "NeuroKit error: _rsp_rrv_formatinput(): Wrong input, "
                    "we couldn't extract rsp_rate and respiratory troughs indices."
                )
            else:
                rsp_rate = signal_rate(
                    df[cols], sampling_rate=sampling_rate, desired_length=len(df)
                )
        else:
            rsp_rate = df[cols[0]].values

    if troughs is None:
        try:
            troughs = _signal_formatpeaks_sanitize(df, key="RSP_Troughs")
        except NameError as e:
            raise ValueError(
                "NeuroKit error: _rsp_rrv_formatinput(): "
                "Wrong input, we couldn't extract "
                "respiratory troughs indices."
            ) from e
    else:
        troughs = _signal_formatpeaks_sanitize(troughs, key="RSP_Troughs")

    return rsp_rate, troughs


def _rsp_rrv_plot(bbi):
    # Axes
    ax1 = bbi[:-1]
    ax2 = bbi[1:]

    # Compute features
    poincare_features = _rsp_rrv_nonlinear(bbi)
    sd1 = poincare_features["SD1"]
    sd2 = poincare_features["SD2"]
    mean_bbi = np.mean(bbi)

    # Plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plt.title("Poincaré Plot", fontsize=20)
    plt.xlabel("BB_n (s)", fontsize=15)
    plt.ylabel("BB_n+1 (s)", fontsize=15)
    plt.xlim(min(bbi) - 10, max(bbi) + 10)
    plt.ylim(min(bbi) - 10, max(bbi) + 10)
    ax.scatter(ax1, ax2, c="b", s=4)

    # Ellipse plot feature
    ellipse = matplotlib.patches.Ellipse(
        xy=(mean_bbi, mean_bbi),
        width=2 * sd2 + 1,
        height=2 * sd1 + 1,
        angle=45,
        linewidth=2,
        fill=False,
    )
    ax.add_patch(ellipse)
    ellipse = matplotlib.patches.Ellipse(
        xy=(mean_bbi, mean_bbi), width=2 * sd2, height=2 * sd1, angle=45
    )
    ellipse.set_alpha(0.02)
    ellipse.set_facecolor("blue")
    ax.add_patch(ellipse)

    # Arrow plot feature
    sd1_arrow = ax.arrow(
        mean_bbi,
        mean_bbi,
        -sd1 * np.sqrt(2) / 2,
        sd1 * np.sqrt(2) / 2,
        linewidth=3,
        ec="r",
        fc="r",
        label="SD1",
    )
    sd2_arrow = ax.arrow(
        mean_bbi,
        mean_bbi,
        sd2 * np.sqrt(2) / 2,
        sd2 * np.sqrt(2) / 2,
        linewidth=3,
        ec="y",
        fc="y",
        label="SD2",
    )

    plt.legend(handles=[sd1_arrow, sd2_arrow], fontsize=12, loc="best")

    return fig

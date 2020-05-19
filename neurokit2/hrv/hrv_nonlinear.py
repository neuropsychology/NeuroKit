# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches

from ..complexity.entropy_sample import entropy_sample
from .hrv_utils import _hrv_sanitize_input
from .hrv_utils import _hrv_get_rri

def hrv_nonlinear(peaks, sampling_rate=1000, show=False):
    """ Computes nonlinear indices of Heart Rate Variability (HRV).

     See references for details.

    Parameters
    ----------
    peaks : dict
        Samples at which cardiac extrema (i.e., R-peaks, systolic peaks) occur.
        Dictionary returned by ecg_findpeaks, ecg_peaks, ppg_findpeaks, or
        ppg_peaks.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks
        occur. Should be at least twice as high as the highest frequency in vhf.
        By default 1000.
    show : bool, optional
        If True, will return a Poincaré plot, a scattergram, which plots each
        RR interval against the next successive one. The ellipse centers around
        the average RR interval. By default False.

    Returns
    -------
    DataFrame
        Contains non-linear HRV metrics:
        - "*SD1*": SD1 is a measure of the spread of RR intervals on the Poincaré plot perpendicular to the line of identity. It is an index of short-term RR interval fluctuations i.e., beat-to-beat variability.
        - "*SD2*": SD2 is a measure of the spread of RR intervals on the Poincaré plot along the line of identity. It is an index of long-term RR interval fluctuations.
        - "*SD2SD1*": the ratio between short and long term fluctuations of the RR intervals (SD2 divided by SD1).
        - "*CSI*": the Cardiac Sympathetic Index, calculated by dividing the longitudinal variability of the Poincaré plot by its transverse variability.
        - "*CVI*": the Cardiac Vagal Index, equal to the logarithm of the product of longitudinal and transverse variability.
        - "*CSI_Modified*": the modified CSI obtained by dividing the square of the longitudinal variability by its transverse variability. Usually used in seizure research.
        - "*SampEn*": the sample entropy measure of HRV, calculated by `entropy_sample()`.

    See Also
    --------
    ecg_peaks, ppg_peaks, hrv_frequency, hrv_time, hrv_summary

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
    >>> hrv = nk.hrv_nonlinear(peaks, sampling_rate=100, show=True)

    References
    ----------
    - Stein, P. K. (2002). Assessing heart rate variability from real-world
      Holter reports. Cardiac electrophysiology review, 6(3), 239-244.
    - Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate
    variability metrics and norms. Frontiers in public health, 5, 258.
    """
    # Sanitize input
    peaks = _hrv_sanitize_input(peaks)

    # Compute R-R intervals (also referred to as NN) in milliseconds
    rri = _hrv_get_rri(peaks, sampling_rate=sampling_rate, interpolate=False)
    diff_rri = np.diff(rri)

    out = {}  # Initialize empty container for results

    # Poincaré
    sd_heart_period = np.std(diff_rri, ddof=1) ** 2
    out["SD1"] = np.sqrt(sd_heart_period * 0.5)
    out["SD2"] = np.sqrt(2 * sd_heart_period - 0.5 * sd_heart_period)
    out["SD2SD1"] = out["SD2"] / out["SD1"]

    # CSI / CVI
    T = 4 * out["SD1"]
    L = 4 * out["SD2"]
    out["CSI"] = L / T
    out["CVI"] = np.log10(L * T)
    out["CSI_Modified"] = L ** 2 / T

    # Entropy
    out["SampEn"] = entropy_sample(rri, dimension=2,
                                   r=0.2 * np.std(rri, ddof=1))

    if show:
        _hrv_nonlinear_show(rri, out)

    out = pd.DataFrame.from_dict(out, orient='index').T.add_prefix("HRV_")
    return out





def _hrv_nonlinear_show(rri, out):

        mean_heart_period = np.mean(rri)
        sd1 = out["SD1"]
        sd2 = out["SD2"]

        # Axes
        ax1 = rri[:-1]
        ax2 = rri[1:]

        # Plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        plt.title("Poincaré Plot", fontsize=20)
        plt.xlabel(r'$RR_{n} (ms)$', fontsize=15)
        plt.ylabel(r'$RR_{n+1} (ms)$', fontsize=15)
        plt.xlim(min(rri) - 10, max(rri) + 10)
        plt.ylim(min(rri) - 10, max(rri) + 10)
        ax.scatter(ax1, ax2, c='#2196F3', s=4)

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
        ellipse.set_facecolor("#2196F3")
        ax.add_patch(ellipse)

        # Arrow plot feature
        sd1_arrow = ax.arrow(mean_heart_period, mean_heart_period,
                             -sd1 * np.sqrt(2) / 2, sd1 * np.sqrt(2) / 2,
                             linewidth=3, ec='#E91E63', fc="#E91E63", label="SD1")
        sd2_arrow = ax.arrow(mean_heart_period,
                             mean_heart_period, sd2 * np.sqrt(2) / 2,
                             sd2 * np.sqrt(2) / 2,
                             linewidth=3, ec='#FF9800', fc="#FF9800", label="SD2")

        plt.legend(handles=[sd1_arrow, sd2_arrow], fontsize=12, loc="best")

        return fig

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from ..complexity.entropy_sample import entropy_sample


def hrv_nonlinear(peaks, sampling_rate=1000, show=False):
    """[summary]

    Parameters
    ----------
    peaks : [type]
        Samples at which cardiac extrema (R-peaks, systolic peaks) occur.
    sampling_rate : int, optional
        Sampling rate of the continuous cardiac signal in which the peaks occur.
        Should be at least twice as high as the highest frequency in vhf. By
        default 1000.
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
    """

    # Compute heart period in milliseconds.
    heart_period = np.diff(peaks) / sampling_rate * 1000
    
    diff_heart_period = np.diff(heart_period)
    
    out = {}

    # Poincaré
    sd_heart_period = np.std(diff_heart_period, ddof=1) ** 2
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
    out["SampEn"] = entropy_sample(heart_period, dimension=2,
                                   r=0.2 * np.std(heart_period, ddof=1))

    if show:
        _show(heart_period, out)
    
    return out


def _show(heart_period, out):
    
        mean_heart_period = np.mean(heart_period)
        sd1 = out["SD1"]
        sd2 = out["SD2"]
        
        # Axes
        ax1 = heart_period[:-1]
        ax2 = heart_period[1:]

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
    
# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from ..complexity.entropy_sample import entropy_sample
from .hrv_utils import _hrv_get_rri, _hrv_sanitize_input


def hrv_nonlinear(peaks, sampling_rate=1000, show=False):
    """
    Computes nonlinear indices of Heart Rate Variability (HRV).

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
    >>> hrv #doctest: +SKIP

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
    out["SampEn"] = entropy_sample(rri, dimension=2, r=0.2 * np.std(rri, ddof=1))

    if show:
        _hrv_nonlinear_show(rri, out)

    out = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("HRV_")
    return out


def _hrv_nonlinear_show(rri, out, ax=None, ax_marg_x=None, ax_marg_y=None):

    mean_heart_period = np.mean(rri)
    sd1 = out["SD1"]
    sd2 = out["SD2"]
    if isinstance(sd1, pd.Series):
        sd1 = float(sd1)
    if isinstance(sd2, pd.Series):
        sd2 = float(sd2)
    ax1 = rri[:-1]
    ax2 = rri[1:]

    # Poincare values
    ax1 = rri[:-1]
    ax2 = rri[1:]

    # Set grid boundaries
    ax1_lim = (max(ax1) - min(ax1)) / 10
    ax2_lim = (max(ax2) - min(ax2)) / 10
    ax1_min = min(ax1) - ax1_lim
    ax1_max = max(ax1) + ax1_lim
    ax2_min = min(ax2) - ax2_lim
    ax2_max = max(ax2) + ax2_lim

    # Prepare figure
    if ax is None and ax_marg_x is None and ax_marg_y is None:
        gs = matplotlib.gridspec.GridSpec(4, 4)
        fig = plt.figure(figsize=(8, 8))
        ax_marg_x = plt.subplot(gs[0, 0:3])
        ax_marg_y = plt.subplot(gs[1:4, 3])
        ax = plt.subplot(gs[1:4, 0:3])
        gs.update(wspace=0.025, hspace=0.05)  # Reduce spaces
        plt.suptitle("Poincaré Plot")
    else:
        fig = None

    # Create meshgrid
    xx, yy = np.mgrid[ax1_min:ax1_max:100j, ax2_min:ax2_max:100j]

    # Fit Gaussian Kernel
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([ax1, ax2])
    kernel = scipy.stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    cmap = matplotlib.cm.get_cmap("Blues", 10)
    ax.contourf(xx, yy, f, cmap=cmap)
    ax.imshow(np.rot90(f), extent=[ax1_min, ax1_max, ax2_min, ax2_max], aspect="auto")

    # Marginal densities
    ax_marg_x.hist(ax1, bins=int(len(ax1) / 10), density=True, alpha=1, color="#ccdff0", edgecolor="none")
    ax_marg_y.hist(
        ax2,
        bins=int(len(ax2) / 10),
        density=True,
        alpha=1,
        color="#ccdff0",
        edgecolor="none",
        orientation="horizontal",
        zorder=1,
    )
    kde1 = scipy.stats.gaussian_kde(ax1)
    x1_plot = np.linspace(ax1_min, ax1_max, len(ax1))
    x1_dens = kde1.evaluate(x1_plot)

    ax_marg_x.fill(x1_plot, x1_dens, facecolor="none", edgecolor="#1b6aaf", alpha=0.8, linewidth=2)
    kde2 = scipy.stats.gaussian_kde(ax2)
    x2_plot = np.linspace(ax2_min, ax2_max, len(ax2))
    x2_dens = kde2.evaluate(x2_plot)
    ax_marg_y.fill_betweenx(x2_plot, x2_dens, facecolor="none", edgecolor="#1b6aaf", linewidth=2, alpha=0.8, zorder=2)

    # Turn off marginal axes labels
    ax_marg_x.axis("off")
    ax_marg_y.axis("off")

    # Plot ellipse
    angle = 45
    width = 2 * sd2 + 1
    height = 2 * sd1 + 1
    xy = (mean_heart_period, mean_heart_period)
    ellipse = matplotlib.patches.Ellipse(xy=xy, width=width, height=height, angle=angle, linewidth=2, fill=False)
    ellipse.set_alpha(0.5)
    ellipse.set_facecolor("#2196F3")
    ax.add_patch(ellipse)

    # Plot points only outside ellipse
    cos_angle = np.cos(np.radians(180.0 - angle))
    sin_angle = np.sin(np.radians(180.0 - angle))
    xc = ax1 - xy[0]
    yc = ax2 - xy[1]
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle
    rad_cc = (xct ** 2 / (width / 2.0) ** 2) + (yct ** 2 / (height / 2.0) ** 2)

    points = np.where(rad_cc > 1)[0]
    ax.plot(ax1[points], ax2[points], "ro", color="k", alpha=0.5, markersize=4)

    # SD1 and SD2 arrow
    sd1_arrow = ax.arrow(
        mean_heart_period,
        mean_heart_period,
        float(-sd1 * np.sqrt(2) / 2),
        float(sd1 * np.sqrt(2) / 2),
        linewidth=3,
        ec="#E91E63",
        fc="#E91E63",
        label="SD1",
    )
    sd2_arrow = ax.arrow(
        mean_heart_period,
        mean_heart_period,
        float(sd2 * np.sqrt(2) / 2),
        float(sd2 * np.sqrt(2) / 2),
        linewidth=3,
        ec="#FF9800",
        fc="#FF9800",
        label="SD2",
    )

    ax.set_xlabel(r"$RR_{n} (ms)$")
    ax.set_ylabel(r"$RR_{n+1} (ms)$")
    ax.legend(handles=[sd1_arrow, sd2_arrow], fontsize=12, loc="best")

    return fig

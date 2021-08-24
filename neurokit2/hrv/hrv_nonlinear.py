# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from ..complexity import (complexity_lempelziv, entropy_approximate,
                          entropy_fuzzy, entropy_multiscale, entropy_sample,
                          entropy_shannon, fractal_correlation, fractal_dfa,
                          fractal_higuchi, fractal_katz)
from ..misc import NeuroKitWarning, find_consecutive
from ..signal import signal_zerocrossings
from .hrv_utils import _hrv_get_rri, _hrv_sanitize_input


def hrv_nonlinear(peaks, sampling_rate=1000, show=False, **kwargs):
    """Computes nonlinear indices of Heart Rate Variability (HRV).

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
    dfa_windows : list
        A list of tuples containing the number of heartbeats to compute the DFA short term scaling
        exponent, α1 and the long term scaling exponent, α2, respectively.
        Defaults to [[4, 11], [12, None]], where
        α1 is estimated from 4 to 11 heartbeats and α2 is estimated from a larger number of heartbeats,
        i.e., 11 beats and above, based on Acharya et al. (2002).
    show : bool, optional
        If True, will return a Poincaré plot, a scattergram, which plots each RR interval against the
        next successive one. The ellipse centers around the average RR interval. By default False.
    **kwargs : optional
        Other arguments to be passed into `fractal_dfa()` and `fractal_correlation()`.


    Returns
    -------
    DataFrame
        Contains non-linear HRV metrics:

        - **Characteristics of the Poincaré Plot Geometry**:

            - **SD1**: SD1 is a measure of the spread of RR intervals on the Poincaré plot
            perpendicular to the line of identity. It is an index of short-term RR interval
            fluctuations, i.e., beat-to-beat variability. It is equivalent (although on another
            scale) to RMSSD, and therefore it is redundant to report correlations with both
            (Ciccone, 2017).

            - **SD2**: SD2 is a measure of the spread of RR intervals on the Poincaré plot along the
            line of identity. It is an index of long-term RR interval fluctuations.

            - **SD1SD2**: the ratio between short and long term fluctuations of the RR intervals
            (SD1 divided by SD2).

            - **S**: Area of ellipse described by SD1 and SD2 (``pi * SD1 * SD2``). It is
            proportional to *SD1SD2*.

            - **CSI**: The Cardiac Sympathetic Index (Toichi, 1997), calculated by dividing the
            longitudinal variability of the Poincaré plot (``4*SD2``) by its transverse variability (``4*SD1``).

            - **CVI**: The Cardiac Vagal Index (Toichi, 1997), equal to the logarithm of the product of
            longitudinal (``4*SD2``) and transverse variability (``4*SD1``).

            - **CSI_Modified**: The modified CSI (Jeppesen, 2014) obtained by dividing the square of
            the longitudinal variability by its transverse variability.

        - **Indices of Heart Rate Asymmetry (HRA), i.e., asymmetry of the Poincaré plot** (Yan, 2017):

            - **GI**: Guzik's Index, defined as the distance of points above line of identity (LI)
            to LI divided by the distance of all points in Poincaré plot to LI except those that
            are located on LI.

            - **SI**: Slope Index, defined as the phase angle of points above LI divided by the
            phase angle of all points in Poincaré plot except those that are located on LI.

            - **AI**: Area Index, defined as the cumulative area of the sectors corresponding to
            the points that are located above LI divided by the cumulative area of sectors
            corresponding to all points in the Poincaré plot except those that are located on LI.

            - **PI**: Porta's Index, defined as the number of points below LI divided by the total
            number of points in Poincaré plot except those that are located on LI.

            - **SD1d** and **SD1a**: short-term variance of contributions of decelerations
            (prolongations of RR intervals) and accelerations (shortenings of RR intervals),
            respectively (Piskorski,  2011).

            - **C1d** and **C1a**: the contributions of heart rate decelerations and accelerations
            to short-term HRV, respectively (Piskorski,  2011).

            - **SD2d** and **SD2a**: long-term variance of contributions of decelerations
            (prolongations of RR intervals) and accelerations (shortenings of RR intervals),
            respectively (Piskorski,  2011).

            - **C2d** and **C2a**: the contributions of heart rate decelerations and accelerations
            to long-term HRV, respectively (Piskorski,  2011).

            - **SDNNd** and **SDNNa**: total variance of contributions of decelerations
            (prolongations of RR intervals) and accelerations (shortenings of RR intervals),
            respectively (Piskorski,  2011).

            - **Cd** and **Ca**: the total contributions of heart rate decelerations and
            accelerations to HRV.

        - **Indices of Heart Rate Fragmentation** (Costa, 2017):

            - **PIP**: Percentage of inflection points of the RR intervals series.

            - **IALS**: Inverse of the average length of the acceleration/deceleration segments.

            - **PSS**: Percentage of short segments.

            - **PAS**: IPercentage of NN intervals in alternation segments.

        - **Indices of Complexity**:

            - **ApEn**: The approximate entropy measure of HRV, calculated by `entropy_approximate()`.

            - **SampEn**: The sample entropy measure of HRV, calculated by `entropy_sample()`.

            - **ShanEn**: The Shannon entropy measure of HRV, calculated by `entropy_shannon()`.

            - **FuzzyEn**: The fuzzy entropy measure of HRV, calculated by `entropy_fuzzy()`.

            - **MSE**: The multiscale entropy measure of HRV, calculated by `entropy_multiscale()`.

            - **CMSE**: The composite multiscale entropy measure of HRV, calculated by `entropy_multiscale()`.

            - **RCMSE**: The refined composite multiscale entropy measure of HRV, calculated by `entropy_multiscale()`.

            - **CD**: The Correlation Dimension of the HR signal, calculated by `fractal_correlation()`.

            - **HFD**: The Higuchi's Fractal Dimension of the HR signal, calculated by `fractal_higuchi()`.
            kmax is set to "default".

            - **KFD**: The Katz's Fractal Dimension of the HR signal, calculated by `fractal_katz()`.

            - **LZC**: The Lempel-Ziv complexity of the HR signal, calculated by `fractal_lempelziv()`.

            - **DFA_alpha1**: The monofractal detrended fluctuation analysis of the HR signal corresponding
            to short-term correlations, calculated by `fractal_dfa()`.

            - **DFA_alpha2**: The monofractal detrended fluctuation analysis of the HR signal corresponding
            to long-term correlations, calculated by `fractal_dfa()`.

            - **DFA_alpha1_ExpRange**: The multifractal detrended fluctuation analysis of the HR signal
            corresponding to short-term correlations, calculated by `fractal_dfa()`. ExpRange is the range of
            singularity exponents, correspoinding to the width of the singularity spectrum.

            - **DFA_alpha2_ExpRange**: The multifractal detrended fluctuation analysis of the HR signal
            corresponding to long-term correlations, calculated by `fractal_dfa()`. ExpRange is the range of
            singularity exponents, correspoinding to the width of the singularity spectrum.

            - **DFA_alpha1_ExpMean**: Multifractal DFA. ExpMean is the mean of singularity exponents.

            - **DFA_alpha2_ExpMean**: Multifractal DFA. ExpMean is the mean of singularity exponents.

            - **DFA_alpha1_DimRange**: The multifractal detrended fluctuation analysis of the HR signal
            corresponding to short-term correlations, calculated by `fractal_dfa()`. DimRange is the range of
            singularity dimensions, correspoinding to the height of the singularity spectrum.

            - **DFA_alpha2_DimRange**: The multifractal detrended fluctuation analysis of the HR signal
            corresponding to long-term correlations, calculated by `fractal_dfa()`. DimRange is the range of
            singularity dimensions, correspoinding to the height of the singularity spectrum.

            - **DFA_alpha1_DimMean**: Multifractal DFA. Dimmean is the mean of singularity dimensions.

            - **DFA_alpha2_DimMean**: Multifractal DFA. Dimmean is the mean of singularity dimensions.

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
    - Yan, C., Li, P., Ji, L., Yao, L., Karmakar, C., & Liu, C. (2017). Area asymmetry of heart
    rate variability signal. Biomedical engineering online, 16(1), 112.

    - Ciccone, A. B., Siedlik, J. A., Wecht, J. M., Deckert, J. A., Nguyen, N. D., & Weir, J. P.
    (2017). Reminder: RMSSD and SD1 are identical heart rate variability metrics. Muscle & nerve,
    56(4), 674-678.

    - Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms.
    Frontiers in public health, 5, 258.

    - Costa, M. D., Davis, R. B., & Goldberger, A. L. (2017). Heart rate fragmentation: a new
    approach to the analysis of cardiac interbeat interval dynamics. Front. Physiol. 8, 255 (2017).

    - Jeppesen, J., Beniczky, S., Johansen, P., Sidenius, P., & Fuglsang-Frederiksen, A. (2014).
    Using Lorenz plot and Cardiac Sympathetic Index of heart rate variability for detecting seizures
    for patients with epilepsy. In 2014 36th Annual International Conference of the IEEE Engineering
    in Medicine and Biology Society (pp. 4563-4566). IEEE.

    - Piskorski, J., & Guzik, P. (2011). Asymmetric properties of long-term and total heart rate
    variability. Medical & biological engineering & computing, 49(11), 1289-1297.

    - Stein, P. K. (2002). Assessing heart rate variability from real-world Holter reports. Cardiac
    electrophysiology review, 6(3), 239-244.

    - Brennan, M. et al. (2001). Do Existing Measures of Poincaré Plot Geometry Reflect Nonlinear
    Features of Heart Rate Variability?. IEEE Transactions on Biomedical Engineering, 48(11), 1342-1347.

    - Toichi, M., Sugiura, T., Murai, T., & Sengoku, A. (1997). A new method of assessing cardiac
    autonomic function and its comparison with spectral analysis and coefficient of variation of R–R
    interval. Journal of the autonomic nervous system, 62(1-2), 79-84.

    - Acharya, R. U., Lim, C. M., & Joseph, P. (2002). Heart rate variability analysis using
    correlation dimension and detrended fluctuation analysis. Itbm-Rbm, 23(6), 333-339.

    """
    # Sanitize input
    peaks = _hrv_sanitize_input(peaks)
    if isinstance(peaks, tuple):  # Detect actual sampling rate
        peaks, sampling_rate = peaks[0], peaks[1]

    # Compute R-R intervals (also referred to as NN) in milliseconds
    rri = _hrv_get_rri(peaks, sampling_rate=sampling_rate, interpolate=False)

    # Initialize empty container for results
    out = {}

    # Poincaré features (SD1, SD2, etc.)
    out = _hrv_nonlinear_poincare(rri, out)

    # Heart Rate Fragmentation
    out = _hrv_nonlinear_fragmentation(rri, out)

    # Heart Rate Asymmetry
    out = _hrv_nonlinear_poincare_hra(rri, out)

    # DFA
    out = _hrv_dfa(peaks, rri, out, **kwargs)

    # Complexity
    r = 0.2 * np.std(rri, ddof=1)
    out["ApEn"] = entropy_approximate(rri, delay=1, dimension=2, r=r)
    out["SampEn"] = entropy_sample(rri, delay=1, dimension=2, r=r)
    out["ShanEn"] = entropy_shannon(rri)
    out["FuzzyEn"] = entropy_fuzzy(rri, delay=1, dimension=2, r=r)
    out["MSE"] = entropy_multiscale(rri, dimension=2, r=r, composite=False, refined=False)
    out["CMSE"] = entropy_multiscale(rri, dimension=2, r=r, composite=True, refined=False)
    out["RCMSE"] = entropy_multiscale(rri, dimension=2, r=r, composite=True, refined=True)

    out["CD"] = fractal_correlation(rri, delay=1, dimension=2, **kwargs)
    out["HFD"] = fractal_higuchi(rri, **kwargs)
    out["KFD"] = fractal_katz(rri)
    out["LZC"] = complexity_lempelziv(rri, **kwargs)

    if show:
        _hrv_nonlinear_show(rri, out)

    out = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("HRV_")
    return out


# =============================================================================
# Get SD1 and SD2
# =============================================================================
def _hrv_nonlinear_poincare(rri, out):
    """Compute SD1 and SD2.

    - Do existing measures of Poincare plot geometry reflect nonlinear features of heart rate
    variability? - Brennan (2001)

    """

    # HRV and hrvanalysis
    rri_n = rri[:-1]
    rri_plus = rri[1:]
    x1 = (rri_n - rri_plus) / np.sqrt(2)  # Eq.7
    x2 = (rri_n + rri_plus) / np.sqrt(2)
    sd1 = np.std(x1, ddof=1)
    sd2 = np.std(x2, ddof=1)

    out["SD1"] = sd1
    out["SD2"] = sd2

    # SD1 / SD2
    out["SD1SD2"] = sd1 / sd2

    # Area of ellipse described by SD1 and SD2
    out["S"] = np.pi * out["SD1"] * out["SD2"]

    # CSI / CVI
    T = 4 * out["SD1"]
    L = 4 * out["SD2"]
    out["CSI"] = L / T
    out["CVI"] = np.log10(L * T)
    out["CSI_Modified"] = L ** 2 / T

    return out


def _hrv_nonlinear_poincare_hra(rri, out):
    """Heart Rate Asymmetry Indices.

    - Asymmetry of Poincaré plot (or termed as heart rate asymmetry, HRA) - Yan (2017)
    - Asymmetric properties of long-term and total heart rate variability - Piskorski (2011)

    """

    N = len(rri) - 1
    x = rri[:-1]  # rri_n, x-axis
    y = rri[1:]  # rri_plus, y-axis

    diff = y - x
    decelerate_indices = np.where(diff > 0)[0]  # set of points above IL where y > x
    accelerate_indices = np.where(diff < 0)[0]  # set of points below IL where y < x
    nochange_indices = np.where(diff == 0)[0]

    # Distances to centroid line l2
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    dist_l2_all = abs((x - centroid_x) + (y - centroid_y)) / np.sqrt(2)

    # Distances to LI
    dist_all = abs(y - x) / np.sqrt(2)

    # Calculate the angles
    theta_all = abs(np.arctan(1) - np.arctan(y / x))  # phase angle LI - phase angle of i-th point
    # Calculate the radius
    r = np.sqrt(x ** 2 + y ** 2)
    # Sector areas
    S_all = 1 / 2 * theta_all * r ** 2

    # Guzik's Index (GI)
    den_GI = np.sum(dist_all)
    num_GI = np.sum(dist_all[decelerate_indices])
    out["GI"] = (num_GI / den_GI) * 100

    # Slope Index (SI)
    den_SI = np.sum(theta_all)
    num_SI = np.sum(theta_all[decelerate_indices])
    out["SI"] = (num_SI / den_SI) * 100

    # Area Index (AI)
    den_AI = np.sum(S_all)
    num_AI = np.sum(S_all[decelerate_indices])
    out["AI"] = (num_AI / den_AI) * 100

    # Porta's Index (PI)
    m = N - len(nochange_indices)  # all points except those on LI
    b = len(accelerate_indices)  # number of points below LI
    out["PI"] = (b / m) * 100

    # Short-term asymmetry (SD1)
    sd1d = np.sqrt(np.sum(dist_all[decelerate_indices] ** 2) / (N - 1))
    sd1a = np.sqrt(np.sum(dist_all[accelerate_indices] ** 2) / (N - 1))

    sd1I = np.sqrt(sd1d ** 2 + sd1a ** 2)
    out["C1d"] = (sd1d / sd1I) ** 2
    out["C1a"] = (sd1a / sd1I) ** 2
    out["SD1d"] = sd1d  # SD1 deceleration
    out["SD1a"] = sd1a  # SD1 acceleration
    # out["SD1I"] = sd1I  # SD1 based on LI, whereas SD1 is based on centroid line l1

    # Long-term asymmetry (SD2)
    longterm_dec = np.sum(dist_l2_all[decelerate_indices] ** 2) / (N - 1)
    longterm_acc = np.sum(dist_l2_all[accelerate_indices] ** 2) / (N - 1)
    longterm_nodiff = np.sum(dist_l2_all[nochange_indices] ** 2) / (N - 1)

    sd2d = np.sqrt(longterm_dec + 0.5 * longterm_nodiff)
    sd2a = np.sqrt(longterm_acc + 0.5 * longterm_nodiff)

    sd2I = np.sqrt(sd2d ** 2 + sd2a ** 2)
    out["C2d"] = (sd2d / sd2I) ** 2
    out["C2a"] = (sd2a / sd2I) ** 2
    out["SD2d"] = sd2d  # SD2 deceleration
    out["SD2a"] = sd2a  # SD2 acceleration
    # out["SD2I"] = sd2I  # identical with SD2

    # Total asymmerty (SDNN)
    sdnnd = np.sqrt(0.5 * (sd1d ** 2 + sd2d ** 2))  # SDNN deceleration
    sdnna = np.sqrt(0.5 * (sd1a ** 2 + sd2a ** 2))  # SDNN acceleration
    sdnn = np.sqrt(sdnnd ** 2 + sdnna ** 2)  # should be similar to sdnn in hrv_time
    out["Cd"] = (sdnnd / sdnn) ** 2
    out["Ca"] = (sdnna / sdnn) ** 2
    out["SDNNd"] = sdnnd
    out["SDNNa"] = sdnna

    return out


def _hrv_nonlinear_fragmentation(rri, out):
    """Heart Rate Fragmentation Indices - Costa (2017)

    The more fragmented a time series is, the higher the PIP, IALS, PSS, and PAS indices will be.
    """

    diff_rri = np.diff(rri)
    zerocrossings = signal_zerocrossings(diff_rri)

    # Percentage of inflection points (PIP)
    out["PIP"] = len(zerocrossings) / len(rri)

    # Inverse of the average length of the acceleration/deceleration segments (IALS)
    accelerations = np.where(diff_rri > 0)[0]
    decelerations = np.where(diff_rri < 0)[0]
    consecutive = find_consecutive(accelerations) + find_consecutive(decelerations)
    lengths = [len(i) for i in consecutive]
    out["IALS"] = 1 / np.average(lengths)

    # Percentage of short segments (PSS) - The complement of the percentage of NN intervals in
    # acceleration and deceleration segments with three or more NN intervals
    out["PSS"] = np.sum(np.asarray(lengths) < 3) / len(lengths)

    # Percentage of NN intervals in alternation segments (PAS). An alternation segment is a sequence
    # of at least four NN intervals, for which heart rate acceleration changes sign every beat. We note
    # that PAS quantifies the amount of a particular sub-type of fragmentation (alternation). A time
    # series may be highly fragmented and have a small amount of alternation. However, all time series
    # with large amount of alternation are highly fragmented.
    alternations = find_consecutive(zerocrossings)
    lengths = [len(i) for i in alternations]
    out["PAS"] = np.sum(np.asarray(lengths) >= 4) / len(lengths)

    return out


# =============================================================================
# DFA
# =============================================================================
def _hrv_dfa(peaks, rri, out, n_windows="default", **kwargs):

    if 'dfa_windows' in kwargs:
        dfa_windows = kwargs['dfa_windows']
    else:
        dfa_windows = [(4, 11), (12, None)]

    # Determine max beats
    if dfa_windows[1][1] is None:
        max_beats = len(peaks) / 10
    else:
        max_beats = dfa_windows[1][1]

    # No. of windows to compute for short and long term
    if n_windows == "default":
        n_windows_short = int(dfa_windows[0][1] - dfa_windows[0][0] + 1)
        n_windows_long = int(max_beats - dfa_windows[1][0] + 1)
    elif isinstance(n_windows, list):
        n_windows_short = n_windows[0]
        n_windows_long = n_windows[1]

    short_window = np.linspace(dfa_windows[0][0], dfa_windows[0][1], n_windows_short).astype(int)
    long_window = np.linspace(dfa_windows[1][0], int(max_beats), n_windows_long).astype(int)

    # For monofractal
    out["DFA_alpha1"] = fractal_dfa(rri, multifractal=False, windows=short_window, **kwargs)['slopes'][0]
    # For multifractal
    mdfa_alpha1 = fractal_dfa(rri,
                              multifractal=True,
                              q=np.arange(-5, 6),
                              windows=short_window, **kwargs)
    out["DFA_alpha1_ExpRange"] = mdfa_alpha1['ExpRange']
    out["DFA_alpha1_ExpMean"] = mdfa_alpha1['ExpMean']
    out["DFA_alpha1_DimRange"] = mdfa_alpha1['DimRange']
    out["DFA_alpha1_DimMean"] = mdfa_alpha1['DimMean']

    # sanatize max_beats
    if max_beats < dfa_windows[1][0] + 1:
        warn(
                "DFA_alpha2 related indices will not be calculated. "
                "The maximum duration of the windows provided for the long-term correlation is smaller "
                "than the minimum duration of windows. Refer to the `windows` argument in `nk.fractal_dfa()` "
                "for more information.",
                category=NeuroKitWarning
            )
        return out
    else:
        # For monofractal
        out["DFA_alpha2"] = fractal_dfa(rri, multifractal=False, windows=long_window, **kwargs)['slopes'][0]
        # For multifractal
        mdfa_alpha2 = fractal_dfa(rri,
                                  multifractal=True,
                                  q=np.arange(-5, 6),
                                  windows=long_window, **kwargs)
        out["DFA_alpha2_ExpRange"] = mdfa_alpha2['ExpRange']
        out["DFA_alpha2_ExpMean"] = mdfa_alpha2['ExpMean']
        out["DFA_alpha2_DimRange"] = mdfa_alpha2['DimRange']
        out["DFA_alpha2_DimMean"] = mdfa_alpha2['DimMean']

    return out

# =============================================================================
# Plot
# =============================================================================
def _hrv_nonlinear_show(rri, out, ax=None, ax_marg_x=None, ax_marg_y=None):

    mean_heart_period = np.mean(rri)
    sd1 = out["SD1"]
    sd2 = out["SD2"]
    if isinstance(sd1, pd.Series):
        sd1 = float(sd1)
    if isinstance(sd2, pd.Series):
        sd2 = float(sd2)

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

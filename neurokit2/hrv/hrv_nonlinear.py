# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from ..complexity import (
    complexity_lempelziv,
    entropy_approximate,
    entropy_fuzzy,
    entropy_multiscale,
    entropy_sample,
    entropy_shannon,
    fractal_correlation,
    fractal_dfa,
    fractal_higuchi,
    fractal_katz,
)
from ..misc import NeuroKitWarning, find_consecutive
from ..signal import signal_zerocrossings
from .hrv_utils import _hrv_format_input
from .intervals_utils import _intervals_successive


def hrv_nonlinear(peaks, sampling_rate=1000, show=False, **kwargs):
    """**Nonlinear indices of Heart Rate Variability (HRV)**

    This function computes non-linear indices, which include features derived from the *Poincaré
    plot*, as well as other :func:`.complexity` indices corresponding to entropy or fractal
    dimension.

    .. hint::
        There exist many more complexity indices available in NeuroKit2, that could be applied to
        HRV. The :func:`.hrv_nonlinear` function only includes the most commonly used indices.
        Please see the documentation page for all the func:`.complexity` features.

    The **Poincaré plot** is a graphical representation of each NN interval plotted against its
    preceding NN interval. The ellipse that emerges is a visual quantification of the correlation
    between successive NN intervals.

    Basic indices derived from the Poincaré plot analysis include:

    * **SD1**: Standard deviation perpendicular to the line of identity. It is an index of
      short-term RR interval fluctuations, i.e., beat-to-beat variability. It is equivalent
      (although on another scale) to RMSSD, and therefore it is redundant to report correlation
      with both.
    * **SD2**: Standard deviation along the identity line. Index of long-term HRV changes.
    * **SD1/SD2**: ratio of *SD1* to *SD2*. Describes the ratio of short term to long term
      variations in HRV.
    * **S**: Area of ellipse described by *SD1* and *SD2* (``pi * SD1 * SD2``). It is
      proportional to *SD1SD2*.
    * **CSI**: The Cardiac Sympathetic Index (Toichi, 1997) is a measure of cardiac sympathetic
      function independent of vagal activity, calculated by dividing the longitudinal variability of
      the Poincaré plot (``4*SD2``) by its transverse variability (``4*SD1``).
    * **CVI**: The Cardiac Vagal Index (Toichi, 1997) is an index of cardiac parasympathetic
      function (vagal activity unaffected by sympathetic activity), and is equal equal to the
      logarithm of the product of longitudinal (``4*SD2``) and transverse variability (``4*SD1``).
    * **CSI_Modified**: The modified CSI (Jeppesen, 2014) obtained by dividing the square of the
      longitudinal variability by its transverse variability.

    Indices of **Heart Rate Asymmetry** (HRA), i.e., asymmetry of the Poincaré plot (Yan, 2017),
    include:

    * **GI**: Guzik's Index, defined as the distance of points above line of identity (LI) to LI
      divided by the distance of all points in Poincaré plot to LI except those that are located on
      LI.
    * **SI**: Slope Index, defined as the phase angle of points above LI divided by the phase angle
      of all points in Poincaré plot except those that are located on LI.
    * **AI**: Area Index, defined as the cumulative area of the sectors corresponding to the points
      that are located above LI divided by the cumulative area of sectors corresponding to all
      points in the Poincaré plot except those that are located on LI.
    * **PI**: Porta's Index, defined as the number of points below LI divided by the total number
      of points in Poincaré plot except those that are located on LI.
    * **SD1d** and **SD1a**: short-term variance of contributions of decelerations (prolongations
      of RR intervals) and accelerations (shortenings of RR intervals), respectively (Piskorski,
      2011)
    * **C1d** and **C1a**: the contributions of heart rate decelerations and accelerations to
      short-term HRV, respectively (Piskorski,  2011).
    * **SD2d** and **SD2a**: long-term variance of contributions of decelerations (prolongations of
      RR intervals) and accelerations (shortenings of RR intervals), respectively (Piskorski, 2011).
    * **C2d** and **C2a**: the contributions of heart rate decelerations and accelerations to
      long-term HRV, respectively (Piskorski,  2011).
    * **SDNNd** and **SDNNa**: total variance of contributions of decelerations (prolongations of
      RR intervals) and accelerations (shortenings of RR intervals), respectively (Piskorski, 2011).
    * **Cd** and **Ca**: the total contributions of heart rate decelerations and accelerations to
      HRV.

    Indices of **Heart Rate Fragmentation** (Costa, 2017) include:

    * **PIP**: Percentage of inflection points of the RR intervals series.
    * **IALS**: Inverse of the average length of the acceleration/deceleration segments.
    * **PSS**: Percentage of short segments.
    * **PAS**: Percentage of NN intervals in alternation segments.

    Indices of **Complexity** and **Fractal Physiology** include:

    * **ApEn**: See :func:`.entropy_approximate`.
    * **SampEn**: See :func:`.entropy_sample`.
    * **ShanEn**: See :func:`.entropy_shannon`.
    * **FuzzyEn**: See :func:`.entropy_fuzzy`.
    * **MSEn**: See :func:`.entropy_multiscale`.
    * **CMSEn**: See :func:`.entropy_multiscale`.
    * **RCMSEn**: See :func:`.entropy_multiscale`.
    * **CD**: See :func:`.fractal_correlation`.
    * **HFD**: See :func:`.fractal_higuchi` (with ``kmax`` set to ``"default"``).
    * **KFD**: See :func:`.fractal_katz`.
    * **LZC**: See :func:`.complexity_lempelziv`.
    * **DFA_alpha1**: The monofractal detrended fluctuation analysis of the HR signal,
      corresponding to short-term correlations. See :func:`.fractal_dfa`.
    * **DFA_alpha2**: The monofractal detrended fluctuation analysis of the HR signal,
      corresponding to long-term correlations. See :func:`.fractal_dfa`.
    * **MFDFA indices**: Indices related to the :func:`multifractal spectrum <.fractal_dfa()>`.

    Other non-linear indices include those based on Recurrence Quantification Analysis (RQA), but
    are not implemented yet (but soon).

    .. tip::
        We strongly recommend checking our open-access paper `Pham et al. (2021)
        <https://doi.org/10.3390/s21123998>`_ on HRV indices, as well as `Lau et al. (2021)
        <https://psyarxiv.com/f8k3x/>`_ on complexity, for more information.

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
    show : bool, optional
        If ``True``, will return a Poincaré plot, a scattergram, which plots each RR interval
        against the next successive one. The ellipse centers around the average RR interval. By
        default ``False``.
    **kwargs
        Other arguments to be passed into :func:`.fractal_dfa` and :func:`.fractal_correlation`.


    Returns
    -------
    DataFrame
        DataFrame consisting of the computed non-linear HRV metrics, which includes:

        .. codebookadd::
            HRV_SD1|Standard deviation perpendicular to the line of identity. It is an index of \
                short-term RR interval fluctuations, i.e., beat-to-beat variability. It is \
                equivalent (although on another scale) to RMSSD, and therefore it is redundant to \
                report correlation with both.
            HRV_SD2|Standard deviation along the identity line. Index of long-term HRV changes.
            HRV_SD1SD2|Ratio of SD1 to SD2. Describes the ratio of short term to long term \
                variations in HRV.
            HRV_S|Area of ellipse described by *SD1* and *SD2* (``pi * SD1 * SD2``). It is \
                proportional to *SD1SD2*.
            HRV_CSI|The Cardiac Sympathetic Index (Toichi, 1997) is a measure of cardiac \
                sympathetic function independent of vagal activity, calculated by dividing the \
                longitudinal variability of the Poincaré plot (``4*SD2``) by its transverse \
                variability (``4*SD1``).
            HRV_CVI|The Cardiac Vagal Index (Toichi, 1997) is an index of cardiac parasympathetic \
                function (vagal activity unaffected by sympathetic activity), and is equal equal \
                to the logarithm of the product of longitudinal (``4*SD2``) and transverse \
                variability (``4*SD1``).
            HRV_CSI_Modified|The modified CSI (Jeppesen, 2014) obtained by dividing the square of \
                the longitudinal variability by its transverse variability.
            HRV_GI|Guzik's Index, defined as the distance of points above line of identity (LI) to \
                LI divided by the distance of all points in Poincaré plot to LI except those that \
                are located on LI.
            HRV_SI|Slope Index, defined as the phase angle of points above LI divided by the phase \
                angle of all points in Poincaré plot except those that are located on LI.
            HRV_AI|Area Index, defined as the cumulative area of the sectors corresponding to the \
                points that are located above LI divided by the cumulative area of sectors \
                corresponding to all points in the Poincaré plot except those that are located \
                on LI.
            HRV_PI|Porta's Index, defined as the number of points below LI divided by the total \
                number of points in Poincaré plot except those that are located on LI.
            HRV_SD1a|Short-term variance of contributions of decelerations (prolongations of RR \
                intervals), (Piskorski, 2011).
            HRV_SD1d|Short-term variance of contributions of accelerations (shortenings of RR \
                intervals), (Piskorski, 2011).
            HRV_C1a|The contributions of heart rate accelerations to short-term HRV, (Piskorski, 2011).
            HRV_C1d|The contributions of heart rate decelerations to short-term HRV, (Piskorski, 2011).
            HRV_SD2a|Long-term variance of contributions of accelerations (shortenings of RR \
                intervals), (Piskorski, 2011).
            HRV_SD2d|Long-term variance of contributions of decelerations (prolongations of RR \
                intervals),  (Piskorski, 2011).
            HRV_C2a|The contributions of heart rate accelerations to long-term HRV, (Piskorski, 2011).
            HRV_C2d|The contributions of heart rate decelerations to long-term HRV, (Piskorski, 2011).
            HRV_SDNNa|Total variance of contributions of accelerations (shortenings of RR \
                intervals), (Piskorski, 2011).
            HRV_SDNNd|Total variance of contributions of decelerations (prolongations of RR \
                intervals), (Piskorski, 2011).
            HRV_Ca|The total contributions of heart rate accelerations to HRV.
            HRV_Cd|The total contributions of heart rate decelerations to HRV.
            HRV_PIP|Percentage of inflection points of the RR intervals series.
            HRV_IALS|Inverse of the average length of the acceleration/deceleration segments.
            HRV_PSS|Percentage of short segments.
            HRV_PAS|Percentage of NN intervals in alternation segments.


    See Also
    --------
    ecg_peaks, ppg_peaks, hrv_frequency, hrv_time, hrv_summary

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Download data
      data = nk.data("bio_resting_5min_100hz")

      # Find peaks
      peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)

      # Compute HRV indices
      @savefig p_hrv_nonlinear1.png scale=100%
      hrv = nk.hrv_nonlinear(peaks, sampling_rate=100, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      hrv


    References
    ----------
    * Pham, T., Lau, Z. J., Chen, S. H., & Makowski, D. (2021). Heart Rate Variability in
      Psychology: A Review of HRV Indices and an Analysis Tutorial. Sensors, 21(12), 3998.
      https://doi.org/10.3390/s21123998
    * Yan, C., Li, P., Ji, L., Yao, L., Karmakar, C., & Liu, C. (2017). Area asymmetry of heart
      rate variability signal. Biomedical engineering online, 16(1), 112.
    * Ciccone, A. B., Siedlik, J. A., Wecht, J. M., Deckert, J. A., Nguyen, N. D., & Weir, J. P.\
      (2017). Reminder: RMSSD and SD1 are identical heart rate variability metrics. Muscle & nerve,
      56 (4), 674-678.
    * Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and
      norms. Frontiers in public health, 5, 258.
    * Costa, M. D., Davis, R. B., & Goldberger, A. L. (2017). Heart rate fragmentation: a new
      approach to the analysis of cardiac interbeat interval dynamics. Front. Physiol. 8, 255.
    * Jeppesen, J., Beniczky, S., Johansen, P., Sidenius, P., & Fuglsang-Frederiksen, A. (2014).
      Using Lorenz plot and Cardiac Sympathetic Index of heart rate variability for detecting
      seizures for patients with epilepsy. In 2014 36th Annual International Conference of the IEE
      Engineering in Medicine and Biology Society (pp. 4563-4566). IEEE.
    * Piskorski, J., & Guzik, P. (2011). Asymmetric properties of long-term and total heart rate
      variability. Medical & biological engineering & computing, 49(11), 1289-1297.
    * Stein, P. K. (2002). Assessing heart rate variability from real-world Holter reports. Cardiac
      electrophysiology review, 6(3), 239-244.
    * Brennan, M. et al. (2001). Do Existing Measures of Poincaré Plot Geometry Reflect Nonlinear
      Features of Heart Rate Variability?. IEEE Transactions on Biomedical Engineering, 48(11),
      1342-1347.
    * Toichi, M., Sugiura, T., Murai, T., & Sengoku, A. (1997). A new method of assessing cardiac
      autonomic function and its comparison with spectral analysis and coefficient of variation of
      R-R interval. Journal of the autonomic nervous system, 62(1-2), 79-84.
    * Acharya, R. U., Lim, C. M., & Joseph, P. (2002). Heart rate variability analysis using
      correlation dimension and detrended fluctuation analysis. Itbm-Rbm, 23(6), 333-339.

    """
    # Sanitize input
    # If given peaks, compute R-R intervals (also referred to as NN) in milliseconds
    rri, rri_time, rri_missing = _hrv_format_input(peaks, sampling_rate=sampling_rate)

    if rri_missing:
        warn(
            "Missing interbeat intervals have been detected. "
            "Note that missing intervals can distort some HRV features, in particular "
            "nonlinear indices.",
            category=NeuroKitWarning,
        )
    # Initialize empty container for results
    out = {}

    # Poincaré features (SD1, SD2, etc.)
    out = _hrv_nonlinear_poincare(rri, rri_time=rri_time, rri_missing=rri_missing, out=out)

    # Heart Rate Fragmentation
    out = _hrv_nonlinear_fragmentation(rri, rri_time=rri_time, rri_missing=rri_missing, out=out)

    # Heart Rate Asymmetry
    out = _hrv_nonlinear_poincare_hra(rri, rri_time=rri_time, rri_missing=rri_missing, out=out)

    # DFA
    out = _hrv_dfa(rri, out, **kwargs)

    # Complexity
    tolerance = 0.2 * np.std(rri, ddof=1)
    out["ApEn"], _ = entropy_approximate(rri, delay=1, dimension=2, tolerance=tolerance)
    out["SampEn"], _ = entropy_sample(rri, delay=1, dimension=2, tolerance=tolerance)
    out["ShanEn"], _ = entropy_shannon(rri)
    out["FuzzyEn"], _ = entropy_fuzzy(rri, delay=1, dimension=2, tolerance=tolerance)
    out["MSEn"], _ = entropy_multiscale(rri, dimension=2, tolerance=tolerance, method="MSEn")
    out["CMSEn"], _ = entropy_multiscale(rri, dimension=2, tolerance=tolerance, method="CMSEn")
    out["RCMSEn"], _ = entropy_multiscale(rri, dimension=2, tolerance=tolerance, method="RCMSEn")

    out["CD"], _ = fractal_correlation(rri, delay=1, dimension=2, **kwargs)
    out["HFD"], _ = fractal_higuchi(rri, k_max=10, **kwargs)
    out["KFD"], _ = fractal_katz(rri)
    out["LZC"], _ = complexity_lempelziv(rri, **kwargs)

    if show:
        _hrv_nonlinear_show(rri, rri_time=rri_time, rri_missing=rri_missing, out=out)

    out = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("HRV_")
    return out


# =============================================================================
# Get SD1 and SD2
# =============================================================================
def _hrv_nonlinear_poincare(rri, rri_time=None, rri_missing=False, out={}):
    """Compute SD1 and SD2.

    - Brennan (2001). Do existing measures of Poincare plot geometry reflect nonlinear features of
      heart rate variability?

    """

    # HRV and hrvanalysis
    rri_n = rri[:-1]
    rri_plus = rri[1:]

    if rri_missing:
        # Only include successive differences
        rri_plus = rri_plus[_intervals_successive(rri, intervals_time=rri_time)]
        rri_n = rri_n[_intervals_successive(rri, intervals_time=rri_time)]

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
    out["CSI_Modified"] = L**2 / T

    return out


def _hrv_nonlinear_poincare_hra(rri, rri_time=None, rri_missing=False, out={}):
    """Heart Rate Asymmetry Indices.

    - Asymmetry of Poincaré plot (or termed as heart rate asymmetry, HRA) - Yan (2017)
    - Asymmetric properties of long-term and total heart rate variability - Piskorski (2011)

    """

    N = len(rri) - 1
    x = rri[:-1]  # rri_n, x-axis
    y = rri[1:]  # rri_plus, y-axis

    if rri_missing:
        # Only include successive differences
        x = x[_intervals_successive(rri, intervals_time=rri_time)]
        y = y[_intervals_successive(rri, intervals_time=rri_time)]
        N = len(x)

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
    r = np.sqrt(x**2 + y**2)
    # Sector areas
    S_all = 1 / 2 * theta_all * r**2

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

    sd1I = np.sqrt(sd1d**2 + sd1a**2)
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

    sd2I = np.sqrt(sd2d**2 + sd2a**2)
    out["C2d"] = (sd2d / sd2I) ** 2
    out["C2a"] = (sd2a / sd2I) ** 2
    out["SD2d"] = sd2d  # SD2 deceleration
    out["SD2a"] = sd2a  # SD2 acceleration
    # out["SD2I"] = sd2I  # identical with SD2

    # Total asymmerty (SDNN)
    sdnnd = np.sqrt(0.5 * (sd1d**2 + sd2d**2))  # SDNN deceleration
    sdnna = np.sqrt(0.5 * (sd1a**2 + sd2a**2))  # SDNN acceleration
    sdnn = np.sqrt(sdnnd**2 + sdnna**2)  # should be similar to sdnn in hrv_time
    out["Cd"] = (sdnnd / sdnn) ** 2
    out["Ca"] = (sdnna / sdnn) ** 2
    out["SDNNd"] = sdnnd
    out["SDNNa"] = sdnna

    return out


def _hrv_nonlinear_fragmentation(rri, rri_time=None, rri_missing=False, out={}):
    """Heart Rate Fragmentation Indices - Costa (2017)

    The more fragmented a time series is, the higher the PIP, IALS, PSS, and PAS indices will be.
    """

    diff_rri = np.diff(rri)
    if rri_missing:
        # Only include successive differences
        diff_rri = diff_rri[_intervals_successive(rri, intervals_time=rri_time)]

    zerocrossings = signal_zerocrossings(diff_rri)

    # Percentage of inflection points (PIP)
    N = len(diff_rri) + 1
    out["PIP"] = len(zerocrossings) / N

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
def _hrv_dfa(rri, out, n_windows="default", **kwargs):

    # if "dfa_windows" in kwargs:
    #    dfa_windows = kwargs["dfa_windows"]
    # else:
    # dfa_windows = [(4, 11), (12, None)]
    # consider using dict.get() mthd directly

    # If the signal is too short, skip it
    if len(rri) < 12:
        out['DFA_alpha1'] = np.nan
        out['DFA_alpha2'] = np.nan
        return out

    dfa_windows = kwargs.get("dfa_windows", [(4, 11), (12, None)])

    # Determine max beats
    if dfa_windows[1][1] is None:
        max_beats = (len(rri) + 1) / 10  # Number of peaks divided by 10
    else:
        max_beats = dfa_windows[1][1]

    # No. of windows to compute for short and long term
    if n_windows == "default":
        n_windows_short = int(dfa_windows[0][1] - dfa_windows[0][0] + 1)
        n_windows_long = int(max_beats - dfa_windows[1][0] + 1)
    elif isinstance(n_windows, list):
        n_windows_short = n_windows[0]
        n_windows_long = n_windows[1]

    # Compute DFA alpha1
    short_window = np.linspace(dfa_windows[0][0], dfa_windows[0][1], n_windows_short).astype(int)
    # For monofractal
    out["DFA_alpha1"], _ = fractal_dfa(rri, multifractal=False, scale=short_window, **kwargs)
    # For multifractal
    mdfa_alpha1, _ = fractal_dfa(rri, multifractal=True, q=np.arange(-5, 6), scale=short_window, **kwargs)
    for k in mdfa_alpha1.columns:
        out["MFDFA_alpha1_" + k] = mdfa_alpha1[k].values[0]

    # Compute DFA alpha2
    # sanatize max_beats
    if max_beats < dfa_windows[1][0] + 1:
        warn(
            "DFA_alpha2 related indices will not be calculated. "
            "The maximum duration of the windows provided for the long-term correlation is smaller "
            "than the minimum duration of windows. Refer to the `scale` argument in `nk.fractal_dfa()` "
            "for more information.",
            category=NeuroKitWarning,
        )
        return out
    else:
        long_window = np.linspace(dfa_windows[1][0], int(max_beats), n_windows_long).astype(int)
        # For monofractal
        out["DFA_alpha2"], _ = fractal_dfa(rri, multifractal=False, scale=long_window, **kwargs)
        # For multifractal
        mdfa_alpha2, _ = fractal_dfa(rri, multifractal=True, q=np.arange(-5, 6), scale=long_window, **kwargs)
        for k in mdfa_alpha2.columns:
            out["MFDFA_alpha2_" + k] = mdfa_alpha2[k].values[0]

    return out


# =============================================================================
# Plot
# =============================================================================
def _hrv_nonlinear_show(rri, rri_time=None, rri_missing=False, out={}, ax=None, ax_marg_x=None, ax_marg_y=None):

    mean_heart_period = np.nanmean(rri)
    sd1 = out["SD1"]
    sd2 = out["SD2"]
    if isinstance(sd1, pd.Series):
        sd1 = float(sd1.iloc[0])
    if isinstance(sd2, pd.Series):
        sd2 = float(sd2.iloc[0])

    # Poincare values
    ax1 = rri[:-1]
    ax2 = rri[1:]

    if rri_missing:
        # Only include successive differences
        ax1 = ax1[_intervals_successive(rri, intervals_time=rri_time)]
        ax2 = ax2[_intervals_successive(rri, intervals_time=rri_time)]

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

    cmap = plt.get_cmap("Blues").resampled(10)
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
    rad_cc = (xct**2 / (width / 2.0) ** 2) + (yct**2 / (height / 2.0) ** 2)

    points = np.where(rad_cc > 1)[0]
    ax.plot(ax1[points], ax2[points], "o", color="k", alpha=0.5, markersize=4)

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

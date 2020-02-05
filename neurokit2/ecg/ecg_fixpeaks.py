# - * - coding: utf-8 - * -
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import scipy.signal
import scipy.stats

from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize


def ecg_fixpeaks(rpeaks, sampling_rate=1000, show=False):
    """Correct R-peaks location based on their interval (RRi).

    Identify erroneous inter-beat-intervals. Lipponen & Tarvainen (2019).

    Parameters
    ----------
    rpeaks : dict
        The samples at which the R-peak occur. Dict returned by
        `ecg_findpeaks()`.
    sampling_rate : int
        The sampling frequency of the signal that contains the peaks (in Hz,
        i.e., samples/second).
    show : bool
        Whether or not to visualize artifacts and artifact thresholds.

    Returns
    -------
    artifacts : dict
        A dictionary containing the indices of artifacts, accessible with the
        keys "etopic", "missed", "extra", and "longshort".

    See Also
    --------
    ecg_clean, ecg_findpeaks, ecg_peaks, ecg_rate, ecg_process, ecg_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>>
    >>> # Get peaks
    >>> ecg_signal = nk.ecg_simulate(20)
    >>> rpeaks = nk.ecg_findpeaks(ecg_signal)
    >>>
    >>> # Add artifacts
    >>> rpeaks["ECG_R_Peaks"] = np.delete(rpeaks["ECG_R_Peaks"], [4, 8])
    >>> artifacts = nk.ecg_fixpeaks(rpeaks, show=True)
    >>> rate_corrected = nk.ecg_rate(rpeaks, artifacts=artifacts,
    >>>                              desired_length=len(ecg))
    >>> rate_uncorrected = nk.ecg_rate(rpeaks, desired_length=len(ecg))
    >>>
    >>> fig, ax = plt.subplots()
    >>> ax.plot(rate_uncorrected, label="heart rate without artifact correction")
    >>> ax.plot(rate_corrected, label="heart rate with artifact correction")
    >>> ax.legend(loc="upper right")

    References
    ----------
    - Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
    heart rate variability time series artefact correction using novel beat
    classification. Journal of medical engineering & technology, 43(3), 173-181.
    10.1080/03091902.2019.1640306

    """
    # Format input.
    rpeaks, desired_length = _signal_formatpeaks_sanitize(rpeaks, desired_length=None)

    artifacts, info = _ecg_fixpeaks_lipponen2019(rpeaks, sampling_rate)
    if show is True:
        _ecg_fixpeaks_lipponen2019_plot(artifacts, info)

    return artifacts






# =============================================================================
# Lipponen & Tarvainen (2019).
# =============================================================================
def _ecg_fixpeaks_lipponen2019_plot(artifacts, info):
    """
    """
    # Extract parameters
    longshort_idcs = artifacts["longshort"]
    ectopic_idcs = artifacts["ectopic"]
    extra_idcs = artifacts["extra"]
    missed_idcs = artifacts["missed"]

    rr = info["rr"]
    drrs = info["drrs"]
    mrrs = info["mrrs"]
    s12 = info["s12"]
    s22 = info["s22"]
    c1 = info["c1"]
    c2 = info["c2"]




    # Visualize artifact type indices.
    fig0, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax0.set_title("Artifact types", fontweight="bold")
    ax0.plot(rr, label="heart period")
    ax0.scatter(longshort_idcs, rr[longshort_idcs], marker='x', c='m',
                s=100, zorder=3, label="long/short")
    ax0.scatter(ectopic_idcs, rr[ectopic_idcs], marker='x', c='g', s=100,
                zorder=3, label="ectopic")
    ax0.scatter(extra_idcs, rr[extra_idcs], marker='x', c='y', s=100,
                zorder=3, label="false positive")
    ax0.scatter(missed_idcs, rr[missed_idcs], marker='x', c='r', s=100,
                zorder=3, label="false negative")
    ax0.legend(loc="upper right")

    # Visualize first threshold.
    ax1.set_title("Consecutive-difference criterion", fontweight="bold")
    ax1.plot(np.abs(drrs), label="difference consecutive heart periods")
    ax1.axhline(1, c='r', label="artifact threshold")
    ax1.legend(loc="upper right")

    ax2.set_title("Difference-from-median criterion", fontweight="bold")
    ax2.plot(np.abs(mrrs), label="difference from median over 11 periods")
    ax2.axhline(3, c="r", label="artifact threshold")
    ax2.legend(loc="upper right")

    # Visualize decision boundaries.
    fig2, (ax3, ax4) = plt.subplots(nrows=1, ncols=2)
    ax3.set_title("Subspace 1", fontweight="bold")
    ax3.set_xlabel("S11")
    ax3.set_ylabel("S12")
    ax3.scatter(drrs, s12, marker="x", label="heart periods")
    verts0 = [(min(drrs), max(s12)),
              (min(drrs), -c1 * min(drrs) + c2),
              (-1, -c1 * -1 + c2),
              (-1, max(s12))]
    poly0 = matplotlib.patches.Polygon(verts0, alpha=0.3, facecolor="r", edgecolor=None,
                                       label="etopic periods")
    ax3.add_patch(poly0)
    verts1 = [(1, -c1 * 1 - c2),
              (1, min(s12)),
              (max(drrs), min(s12)),
              (max(drrs), -c1 * max(drrs) - c2)]
    poly1 = matplotlib.patches.Polygon(verts1, alpha=0.3, facecolor="r", edgecolor=None)
    ax3.add_patch(poly1)
    ax3.legend(loc="upper right")

    ax4.set_title("Subspace 2", fontweight="bold")
    ax4.set_xlabel("S21")
    ax4.set_ylabel("S22")
    ax4.scatter(drrs, s22, marker="x", label="heart periods")
    verts2 = [(min(drrs), max(s22)),
              (min(drrs), 1),
              (-1, 1),
              (-1, max(s22))]
    poly2 = matplotlib.patches.Polygon(verts2, alpha=0.3, facecolor="r", edgecolor=None,
                                       label="short periods")
    ax4.add_patch(poly2)
    verts3 = [(1, -1),
              (1, min(s22)),
              (max(drrs), min(s22)),
              (max(drrs), -1)]
    poly3 = matplotlib.patches.Polygon(verts3, alpha=0.3, facecolor="y", edgecolor=None,
                                       label="long periods")
    ax4.add_patch(poly3)
    ax4.legend(loc="upper right")






def _ecg_fixpeaks_lipponen2019(rpeaks, sampling_rate=1000):

    # Set free parameters.
    c1 = 0.13
    c2 = 0.17
    alpha = 5.2
    window_half = 45
    medfilt_order = 11

    # Compute period series (make sure it has same numer of elements as peaks);
    # peaks are in samples, convert to seconds.
    rr = np.ediff1d(rpeaks, to_begin=0) / sampling_rate
    # For subsequent analysis it is important that the first element has
    # a value in a realistic range (e.g., for median filtering).
    rr[0] = np.mean(rr)

    # Compute differences of consecutive periods.
    drrs = np.ediff1d(rr, to_begin=0)
    drrs[0] = np.mean(drrs)
    # Normalize by threshold.
    drrs, _ = _threshold_normalization(drrs, alpha, window_half)

    # Pad drrs with one element.
    padding = 2
    drrs_pad = np.pad(drrs, padding, "reflect")
    # Cast drrs to two-dimesnional subspace s1.
    s12 = np.zeros(drrs.size)
    for d in np.arange(padding, padding + drrs.size):

        if drrs_pad[d] > 0:
            s12[d - padding] = np.max([drrs_pad[d - 1], drrs_pad[d + 1]])
        elif drrs_pad[d] < 0:
            s12[d - padding] = np.min([drrs_pad[d - 1], drrs_pad[d + 1]])

    # Cast drrs to two-dimensional subspace s2 (looping over d a second
    # consecutive time is choice to be explicit rather than efficient).
    s22 = np.zeros(drrs.size)
    for d in np.arange(padding, padding + drrs.size):

        if drrs_pad[d] > 0:
            s22[d - padding] = np.max([drrs_pad[d + 1], drrs_pad[d + 2]])
        elif drrs_pad[d] < 0:
            s22[d - padding] = np.min([drrs_pad[d + 1], drrs_pad[d + 2]])

    # Compute deviation of RRs from median RRs.
    padding = medfilt_order // 2    # pad RR series before filtering
    rr_pad = np.pad(rr, padding, "reflect")
    medrr = scipy.signal.medfilt(rr_pad, medfilt_order)
    medrr = medrr[padding:padding + rr.size]    # remove padding
    mrrs = rr - medrr
    mrrs[mrrs < 0] = mrrs[mrrs < 0] * 2
    mrrs, th2 = _threshold_normalization(mrrs, alpha, window_half)    # normalize by threshold

    # Artifact identification
    #########################
    # Keep track of indices that need to be interpolated, removed, or added.
    extra_idcs = []
    missed_idcs = []
    ectopic_idcs = []
    longshort_idcs = []

    for i in range(rpeaks.size - 2):

        # Check for etopic peaks.
        if np.abs(drrs[i]) <= 1:
            continue

        # Based on Figure 2a.
        eq1 = np.logical_and(drrs[i] > 1, s12[i] < (-c1 * drrs[i] - c2))
        eq2 = np.logical_and(drrs[i] < -1, s12[i] > (-c1 * drrs[i] + c2))

        if np.any([eq1, eq2]):
            # If any of the two equations is true.
            ectopic_idcs.append(i)
            continue

        # If none of the two equations is true.
        # Based on Figure 2b.
        if np.logical_or(np.abs(drrs[i]) > 1, np.abs(mrrs[i]) > 3):
            # Long beat.
            eq3 = np.logical_and(drrs[i] > 1, s22[i] < -1)
            eq4 = np.abs(mrrs[i]) > 3
            # Short beat.
            eq5 = np.logical_and(drrs[i] < -1, s22[i] > 1)

        if ~np.any([eq3, eq4, eq5]):
            # Of none of the three equations is true: normal beat.
            continue

        # If any of the three equations is true: check for missing or extra
        # peaks.

        # Missing.
        eq6 = np.abs(rr[i] / 2 - medrr[i]) < th2[i]
        # Extra.
        eq7 = np.abs(rr[i] + rr[i + 1] - medrr[i]) < th2[i]

        # Check if short or extra.
        if np.any([eq3, eq4]):
            if eq7:
                extra_idcs.append(i)
            else:
                longshort_idcs.append(i)
                if np.abs(drrs[i + 1]) < np.abs(drrs[i + 2]):
                    longshort_idcs.append(i + 1)
        # Check if long or missing.
        if eq5:
            if eq6:
                missed_idcs.append(i)
            else:
                longshort_idcs.append(i)
                if np.abs(drrs[i + 1]) < np.abs(drrs[i + 2]):
                    longshort_idcs.append(i + 1)

    # Prepare output
    artifacts = {"ectopic": ectopic_idcs, "missed": missed_idcs,
                 "extra": extra_idcs, "longshort": longshort_idcs}

    info = {"rr": rr, "drrs": drrs, "mrrs": mrrs, "c1": c1, "c2": c2, "s12": s12, "s22": s22}

    return artifacts, info






def _threshold_normalization(data, alpha, window_half):
    wh = window_half
    # compute threshold
    th = np.zeros(data.size)
    if data.size <= 2 * wh:
        th[:] = alpha * (scipy.stats.iqr(np.abs(data)) / 2)
        # normalize data by threshold
        data_th = np.divide(data, th)
    else:
        data_pad = np.pad(data, wh, "reflect")
        for i in np.arange(wh, wh + data.size):
            th[i - wh] = alpha * (scipy.stats.iqr(np.abs(data_pad[i - wh:i + wh])) / 2)
        # normalize data by threshold (remove padding)
        data_th = np.divide(data_pad[wh:wh + data.size], th)
    return data_th, th

# - * - coding: utf-8 - * -
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.gridspec as gridspec
import scipy.signal
import scipy.stats


def ecg_fixpeaks(rpeaks, sampling_rate=1000, iterative=True, show=False):
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
    iterative : bool
        Whether or not to apply the artifact correction repeatedly (results
        in superior artifact correction).
    show : bool
        Whether or not to visualize artifacts and artifact thresholds.

    Returns
    -------
    artifacts : dict
        A dictionary containing the indices of artifacts, accessible with the
        keys "ectopic", "missed", "extra", and "longshort".

    See Also
    --------
    ecg_clean, ecg_findpeaks, ecg_peaks, ecg_rate, ecg_process, ecg_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>> import matplotlib.pyplot as plt

    >>> ecg = nk.ecg_simulate(duration=240, noise=0.1, heart_rate=70,
    >>>                       random_state=41)
    >>> rpeaks_uncorrected = nk.ecg_findpeaks(ecg)
    >>> artifacts, rpeaks_corrected = nk.ecg_fixpeaks(rpeaks_uncorrected,
    >>>                                               iterative=True,
    >>>                                               show=True)
    >>> rate_corrected = nk.ecg_rate(rpeaks_uncorrected,
    >>>                              desired_length=len(ecg))
    >>> rate_uncorrected = nk.ecg_rate(rpeaks, desired_length=len(ecg_signal))
    >>>
    >>> fig, ax = plt.subplots()
    >>> ax.plot(rate_uncorrected, label="heart rate without artifact correction")
    >>> ax.plot(rate_corrected, label="heart rate with artifact correction")
    >>> ax.legend(loc="upper right")

    References
    ----------
    - Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart
    rate variability time series artefact correction using novel beat
    classification. Journal of medical engineering & technology, 43(3),
    173-181. 10.1080/03091902.2019.1640306

    """
    # Format input.
    rpeaks = rpeaks["ECG_R_Peaks"]
    artifacts, subspaces = _find_artifacts_lipponen2019(rpeaks, sampling_rate)
    rpeaks_corrected = _fix_artifacts_lipponen2019(rpeaks, artifacts,
                                                   sampling_rate)

    if iterative:
        # Iteratively apply the artifact correction until the number of
        # artifact reaches an equilibrium (i.e., the number of artifacts
        # does not change anymore from one iteration to the next)
        n_artifacts_previous = np.inf
        n_artifacts_current = sum([len(i) for i in artifacts.values()])

        previous_diff = 0

        while n_artifacts_current - n_artifacts_previous != previous_diff:

            previous_diff = n_artifacts_previous - n_artifacts_current

            artifacts, subspaces = _find_artifacts_lipponen2019(rpeaks_corrected,
                                                                sampling_rate)
            rpeaks_corrected = _fix_artifacts_lipponen2019(rpeaks_corrected,
                                                           artifacts,
                                                           sampling_rate)

            n_artifacts_previous = n_artifacts_current
            n_artifacts_current = sum([len(i) for i in artifacts.values()])

    if show:
        _plot_artifacts_lipponen2019(artifacts, subspaces)

    return artifacts, {"ECG_R_Peaks": rpeaks_corrected}


# =============================================================================
# Lipponen & Tarvainen (2019).
# =============================================================================
def _find_artifacts_lipponen2019(rpeaks, sampling_rate=1000):

    # Set fixed parameters.
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
    rr[0] = np.mean(rr[1:])

    # Compute differences of consecutive periods.
    drrs = np.ediff1d(rr, to_begin=0)
    drrs[0] = np.mean(drrs[1:])
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

        # Check for ectopic peaks.
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
        if ~np.any([np.abs(drrs[i]) > 1, np.abs(mrrs[i]) > 3]):
            continue

        # Long beat.
        eq3 = np.logical_and(drrs[i] > 1, s22[i] < -1)
        eq4 = np.abs(mrrs[i]) > 3
        # Short beat.
        eq5 = np.logical_and(drrs[i] < -1, s22[i] > 1)

        if ~np.any([eq3, eq4, eq5]):
            # If none of the three equations is true: normal beat.
            continue

        # If any of the three equations is true: check for missing or extra
        # peaks.

        # Missing.
        eq6 = np.abs(rr[i] / 2 - medrr[i]) < th2[i]
        # Extra.
        eq7 = np.abs(rr[i] + rr[i + 1] - medrr[i]) < th2[i]

        # Check if short or extra.
        if eq5:
            if eq7:
                extra_idcs.append(i)
            else:
                longshort_idcs.append(i)
                if np.abs(drrs[i + 1]) < np.abs(drrs[i + 2]):
                    longshort_idcs.append(i + 1)
        # Check if long or missing.
        if np.any([eq3, eq4]):
            if eq6:
                missed_idcs.append(i)
            else:
                longshort_idcs.append(i)
                if np.abs(drrs[i + 1]) < np.abs(drrs[i + 2]):
                    longshort_idcs.append(i + 1)

    # Prepare output
    artifacts = {"ectopic": ectopic_idcs, "missed": missed_idcs,
                 "extra": extra_idcs, "longshort": longshort_idcs}

    subspaces = {"rr": rr, "drrs": drrs, "mrrs": mrrs, "s12": s12, "s22": s22,
                 "c1": c1, "c2": c2}

    return artifacts, subspaces


def _fix_artifacts_lipponen2019(rpeaks, artifacts, sampling_rate):

    extra_idcs = artifacts["extra"]
    missed_idcs = artifacts["missed"]
    ectopic_idcs = artifacts["ectopic"]
    longshort_idcs = artifacts["longshort"]

    # Delete extra peaks.
    if extra_idcs:
        rpeaks = np.delete(rpeaks, extra_idcs)
        # Update remaining indices.
        missed_idcs = _update_indices(extra_idcs, missed_idcs, -1)
        ectopic_idcs = _update_indices(extra_idcs, ectopic_idcs, -1)
        longshort_idcs = _update_indices(extra_idcs, longshort_idcs, -1)

    # Add missing peaks.
    if missed_idcs:
        # Calculate the position(s) of new beat(s). Make sure to not generate
        # negative indices. prev_peaks and next_peaks must have the same
        # number of elements.
        missed_idcs = np.array(missed_idcs)
        valid_idcs = np.logical_and(missed_idcs > 1, missed_idcs < len(rpeaks))
        missed_idcs = missed_idcs[valid_idcs]
        prev_rpeaks = rpeaks[[i - 1 for i in missed_idcs]]
        next_rpeaks = rpeaks[missed_idcs]
        added_rpeaks = prev_rpeaks + (next_rpeaks - prev_rpeaks) / 2
        # Add the new peaks before the missed indices (see numpy docs).
        rpeaks = np.insert(rpeaks, missed_idcs, added_rpeaks)
        # Update remaining indices.
        ectopic_idcs = _update_indices(missed_idcs, ectopic_idcs, 1)
        longshort_idcs = _update_indices(missed_idcs, longshort_idcs, 1)

    # Interpolate ectopic as well as long or short peaks (important to do
    # this after peaks are deleted and/or added).
    interp_idcs = np.concatenate((ectopic_idcs, longshort_idcs)).astype(int)
    if interp_idcs.size > 0:
        interp_idcs.sort(kind='mergesort')
        # Make sure to not generate negative indices, or indices that exceed
        # the total number of peaks.
        # Make sure to not generate negative indices, or indices that exceed
        # the total number of peaks. prev_peaks and next_peaks must have the
        # same number of elements.
        valid_idcs = np.logical_and(interp_idcs > 1, interp_idcs < len(rpeaks))
        interp_idcs = interp_idcs[valid_idcs]
        prev_rpeaks = rpeaks[[i - 1 for i in interp_idcs]]
        next_rpeaks = rpeaks[[i + 1 for i in interp_idcs]]
        rpeaks_interp = prev_rpeaks + (next_rpeaks - prev_rpeaks) / 2
        # Shift the R-peaks from the old to the new position.
        rpeaks = np.delete(rpeaks, interp_idcs)
        rpeaks = np.concatenate((rpeaks, rpeaks_interp)).astype(int)
        rpeaks.sort(kind="mergesort")
        rpeaks = np.unique(rpeaks)

    return rpeaks


def _plot_artifacts_lipponen2019(artifacts, info):
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

    # Set grids
    gs = matplotlib.gridspec.GridSpec(ncols=4, nrows=3,
                                      width_ratios=[1, 2, 2, 2])
    fig = plt.figure(constrained_layout=False)
    ax0 = fig.add_subplot(gs[0, :-2])
    ax1 = fig.add_subplot(gs[1, :-2])
    ax2 = fig.add_subplot(gs[2, :-2])
    ax3 = fig.add_subplot(gs[:, -1])
    ax4 = fig.add_subplot(gs[:, -2])

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

    # Visualize second threshold.
    ax2.set_title("Difference-from-median criterion", fontweight="bold")
    ax2.plot(np.abs(mrrs), label="difference from median over 11 periods")
    ax2.axhline(3, c="r", label="artifact threshold")
    ax2.legend(loc="upper right")

    # Visualize subspaces.
    ax4.set_title("Subspace 1", fontweight="bold")
    ax4.set_xlabel("S11")
    ax4.set_ylabel("S12")
    ax4.scatter(drrs, s12, marker="x", label="heart periods")
    verts0 = [(min(drrs), max(s12)),
              (min(drrs), -c1 * min(drrs) + c2),
              (-1, -c1 * -1 + c2),
              (-1, max(s12))]
    poly0 = matplotlib.patches.Polygon(verts0, alpha=0.3, facecolor="r",
                                       edgecolor=None, label="ectopic periods")
    ax4.add_patch(poly0)
    verts1 = [(1, -c1 * 1 - c2),
              (1, min(s12)),
              (max(drrs), min(s12)),
              (max(drrs), -c1 * max(drrs) - c2)]
    poly1 = matplotlib.patches.Polygon(verts1, alpha=0.3, facecolor="r",
                                       edgecolor=None)
    ax4.add_patch(poly1)
    ax4.legend(loc="upper right")

    ax3.set_title("Subspace 2", fontweight="bold")
    ax3.set_xlabel("S21")
    ax3.set_ylabel("S22")
    ax3.scatter(drrs, s22, marker="x", label="heart periods")
    verts2 = [(min(drrs), max(s22)),
              (min(drrs), 1),
              (-1, 1),
              (-1, max(s22))]
    poly2 = matplotlib.patches.Polygon(verts2, alpha=0.3, facecolor="r",
                                       edgecolor=None, label="short periods")
    ax3.add_patch(poly2)
    verts3 = [(1, -1),
              (1, min(s22)),
              (max(drrs), min(s22)),
              (max(drrs), -1)]
    poly3 = matplotlib.patches.Polygon(verts3, alpha=0.3, facecolor="y",
                                       edgecolor=None, label="long periods")
    ax3.add_patch(poly3)
    ax3.legend(loc="upper right")


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


def _update_indices(source_idcs, update_idcs, update):
    """
    for every element s in source_idcs, change every element u in update_idcs
    according to update, if u is larger than s
    """
    update_idcs_buffer = update_idcs
    for s in source_idcs:
        # find the indices (of indices) that need to be updated
        updates = [i for i, j in enumerate(update_idcs) if j > s]
        for u in updates:
            update_idcs_buffer[u] += update
    return update_idcs_buffer

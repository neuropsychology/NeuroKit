# - * - coding: utf-8 - * -
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..stats import standardize
from .signal_formatpeaks import _signal_formatpeaks_sanitize
from .signal_period import signal_period


def signal_fixpeaks(
    peaks,
    sampling_rate=1000,
    iterative=True,
    show=False,
    interval_min=None,
    interval_max=None,
    relative_interval_min=None,
    relative_interval_max=None,
    robust=False,
    method="Kubios",
):
    """Correct erroneous peak placements.

    Identify and correct erroneous peak placements based on outliers in peak-to-peak differences (period).

    Parameters
    ----------
    peaks : list or array or DataFrame or Series or dict
        The samples at which the peaks occur. If an array is passed in, it is assumed that it was obtained
        with `signal_findpeaks()`. If a DataFrame is passed in, it is assumed to be obtained with `ecg_findpeaks()`
        or `ppg_findpeaks()` and to be of the same length as the input signal.
    sampling_rate : int
        The sampling frequency of the signal that contains the peaks (in Hz, i.e., samples/second).
    iterative : bool
        Whether or not to apply the artifact correction repeatedly (results in superior artifact correction).
    show : bool
        Whether or not to visualize artifacts and artifact thresholds.
    interval_min : float
        The minimum interval between the peaks.
    interval_max : float
        The maximum interval between the peaks.
    relative_interval_min : float
        The minimum interval between the peaks as relative to the sample (expressed in
        standard deviation from the mean).
    relative_interval_max : float
        The maximum interval between the peaks as relative to the sample (expressed in
        standard deviation from the mean).
    robust : bool
        Use a robust method of standardization (see `standardize()`) for the relative thresholds.
    method : str
        Either "Kubios" or "Neurokit". "Kubios" uses the artifact detection and correction described
        in Lipponen, J. A., & Tarvainen, M. P. (2019). Note that "Kubios" is only meant for peaks in
        ECG or PPG. "neurokit" can be used with peaks in ECG, PPG, or respiratory data.

    Returns
    -------
    peaks_clean : array
        The corrected peak locations.
    artifacts : dict
        Only if method="Kubios". A dictionary containing the indices of artifacts, accessible with the
        keys "ectopic", "missed", "extra", and "longshort".

    See Also
    --------
    signal_findpeaks, ecg_findpeaks, ecg_peaks, ppg_findpeaks, ppg_peaks

    Examples
    --------
    >>> import neurokit2 as nk
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # Kubios
    >>> ecg = nk.ecg_simulate(duration=240, noise=0.25, heart_rate=70, random_state=42)
    >>> rpeaks_uncorrected = nk.ecg_findpeaks(ecg)
    >>> artifacts, rpeaks_corrected = nk.signal_fixpeaks(rpeaks_uncorrected, iterative=True,
    ...                                                  show=True, method="Kubios")
    >>> rate_corrected = nk.signal_rate(rpeaks_corrected, desired_length=len(ecg))
    >>> rate_uncorrected = nk.signal_rate(rpeaks_uncorrected, desired_length=len(ecg))
    >>>
    >>> fig, ax = plt.subplots()
    >>> ax.plot(rate_uncorrected, label="heart rate without artifact correction") #doctest: +SKIP
    >>> ax.plot(rate_corrected, label="heart rate with artifact correction") #doctest: +SKIP
    >>> ax.legend(loc="upper right") #doctest: +SKIP
    >>>
    >>> # NeuroKit
    >>> signal = nk.signal_simulate(duration=4, sampling_rate=1000, frequency=1)
    >>> peaks_true = nk.signal_findpeaks(signal)["Peaks"]
    >>> peaks = np.delete(peaks_true, [1])  # create gaps
    >>>
    >>> signal = nk.signal_simulate(duration=20, sampling_rate=1000, frequency=1)
    >>> peaks_true = nk.signal_findpeaks(signal)["Peaks"]
    >>> peaks = np.delete(peaks_true, [5, 15])  # create gaps
    >>> peaks = np.sort(np.append(peaks, [1350, 11350, 18350]))  # add artifacts
    >>>
    >>> peaks_corrected = nk.signal_fixpeaks(peaks=peaks, interval_min=0.5, interval_max=1.5, method="neurokit")
    >>> # Plot and shift original peaks to the rightto see the difference.
    >>> fig = nk.events_plot([peaks + 50, peaks_corrected], signal)
    >>> fig #doctest: +SKIP

    References
    ----------
    - Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability time
    series artefact correction using novel beat classification. Journal of medical engineering & technology,
    43(3), 173-181. 10.1080/03091902.2019.1640306

    """
    # Format input
    peaks = _signal_formatpeaks_sanitize(peaks)

    # If method Kubios
    if method.lower() == "kubios":

        return _signal_fixpeaks_kubios(peaks, sampling_rate=sampling_rate, iterative=iterative, show=show)

    # Else method is NeuroKit
    return _signal_fixpeaks_neurokit(
        peaks,
        sampling_rate=sampling_rate,
        interval_min=interval_min,
        interval_max=interval_max,
        relative_interval_min=relative_interval_min,
        relative_interval_max=relative_interval_max,
        robust=robust,
    )


# =============================================================================
# Methods
# =============================================================================
def _signal_fixpeaks_neurokit(
    peaks,
    sampling_rate=1000,
    interval_min=None,
    interval_max=None,
    relative_interval_min=None,
    relative_interval_max=None,
    robust=False,
):
    """Neurokit method."""

    peaks_clean = _remove_small(peaks, sampling_rate, interval_min, relative_interval_min, robust)
    peaks_clean = _interpolate_big(peaks, sampling_rate, interval_max, relative_interval_max, robust)

    return peaks_clean


def _signal_fixpeaks_kubios(peaks, sampling_rate=1000, iterative=True, show=False):
    """kubios method."""

    # Get corrected peaks and normal-to-normal intervals.
    artifacts, subspaces = _find_artifacts(peaks, sampling_rate=sampling_rate)
    peaks_clean = _correct_artifacts(artifacts, peaks)

    if iterative:

        # Iteratively apply the artifact correction until the number of artifact
        # reaches an equilibrium (i.e., the number of artifacts does not change
        # anymore from one iteration to the next).
        n_artifacts_previous = np.inf
        n_artifacts_current = sum([len(i) for i in artifacts.values()])

        previous_diff = 0

        while n_artifacts_current - n_artifacts_previous != previous_diff:

            previous_diff = n_artifacts_previous - n_artifacts_current

            artifacts, subspaces = _find_artifacts(peaks_clean, sampling_rate=sampling_rate)
            peaks_clean = _correct_artifacts(artifacts, peaks_clean)

            n_artifacts_previous = n_artifacts_current
            n_artifacts_current = sum([len(i) for i in artifacts.values()])

    if show:
        _plot_artifacts_lipponen2019(artifacts, subspaces)

    return artifacts, peaks_clean


# =============================================================================
# Kubios: Lipponen & Tarvainen (2019).
# =============================================================================
def _find_artifacts(peaks, c1=0.13, c2=0.17, alpha=5.2, window_width=91, medfilt_order=11, sampling_rate=1000):

    # Compute period series (make sure it has same numer of elements as peaks);
    # peaks are in samples, convert to seconds.
    rr = np.ediff1d(peaks, to_begin=0) / sampling_rate
    # For subsequent analysis it is important that the first element has
    # a value in a realistic range (e.g., for median filtering).
    rr[0] = np.mean(rr[1:])

    # Artifact identification #################################################
    ###########################################################################

    # Compute dRRs: time series of differences of consecutive periods (dRRs).
    drrs = np.ediff1d(rr, to_begin=0)
    drrs[0] = np.mean(drrs[1:])
    # Normalize by threshold.
    th1 = _compute_threshold(drrs, alpha, window_width)
    drrs /= th1

    # ignore division by 0 warning
    np.seterr(divide="ignore", invalid="ignore")

    # Cast dRRs to subspace s12.
    # Pad drrs with one element.
    padding = 2
    drrs_pad = np.pad(drrs, padding, "reflect")

    s12 = np.zeros(drrs.size)
    for d in np.arange(padding, padding + drrs.size):

        if drrs_pad[d] > 0:
            s12[d - padding] = np.max([drrs_pad[d - 1], drrs_pad[d + 1]])
        elif drrs_pad[d] < 0:
            s12[d - padding] = np.min([drrs_pad[d - 1], drrs_pad[d + 1]])

    # Cast dRRs to subspace s22.
    s22 = np.zeros(drrs.size)
    for d in np.arange(padding, padding + drrs.size):

        if drrs_pad[d] >= 0:
            s22[d - padding] = np.min([drrs_pad[d + 1], drrs_pad[d + 2]])
        elif drrs_pad[d] < 0:
            s22[d - padding] = np.max([drrs_pad[d + 1], drrs_pad[d + 2]])

    # Compute mRRs: time series of deviation of RRs from median.
    df = pd.DataFrame({"signal": rr})
    medrr = df.rolling(medfilt_order, center=True, min_periods=1).median().signal.to_numpy()
    mrrs = rr - medrr
    mrrs[mrrs < 0] = mrrs[mrrs < 0] * 2
    # Normalize by threshold.
    th2 = _compute_threshold(mrrs, alpha, window_width)
    mrrs /= th2

    # Artifact classification #################################################
    ###########################################################################

    # Artifact classes.
    extra_idcs = []
    missed_idcs = []
    ectopic_idcs = []
    longshort_idcs = []

    i = 0
    while i < rr.size - 2:  # The flow control is implemented based on Figure 1

        if np.abs(drrs[i]) <= 1:  # Figure 1
            i += 1
            continue
        eq1 = np.logical_and(drrs[i] > 1, s12[i] < (-c1 * drrs[i] - c2))  # pylint: disable=E1111
        eq2 = np.logical_and(drrs[i] < -1, s12[i] > (-c1 * drrs[i] + c2))  # pylint: disable=E1111

        if np.any([eq1, eq2]):
            # If any of the two equations is true.
            ectopic_idcs.append(i)
            i += 1
            continue
        # If none of the two equations is true.
        if ~np.any([np.abs(drrs[i]) > 1, np.abs(mrrs[i]) > 3]):  # Figure 1
            i += 1
            continue
        longshort_candidates = [i]
        # Check if the following beat also needs to be evaluated.
        if np.abs(drrs[i + 1]) < np.abs(drrs[i + 2]):
            longshort_candidates.append(i + 1)

        for j in longshort_candidates:
            # Long beat.
            eq3 = np.logical_and(drrs[j] > 1, s22[j] < -1)  # pylint: disable=E1111
            # Long or short.
            eq4 = np.abs(mrrs[j]) > 3  # Figure 1
            # Short beat.
            eq5 = np.logical_and(drrs[j] < -1, s22[j] > 1)  # pylint: disable=E1111

            if ~np.any([eq3, eq4, eq5]):
                # If none of the three equations is true: normal beat.
                i += 1
                continue
            # If any of the three equations is true: check for missing or extra
            # peaks.

            # Missing.
            eq6 = np.abs(rr[j] / 2 - medrr[j]) < th2[j]  # Figure 1
            # Extra.
            eq7 = np.abs(rr[j] + rr[j + 1] - medrr[j]) < th2[j]  # Figure 1

            # Check if extra.
            if np.all([eq5, eq7]):
                extra_idcs.append(j)
                i += 1
                continue
            # Check if missing.
            if np.all([eq3, eq6]):
                missed_idcs.append(j)
                i += 1
                continue
            # If neither classified as extra or missing, classify as "long or
            # short".
            longshort_idcs.append(j)
            i += 1

    # Prepare output
    artifacts = {"ectopic": ectopic_idcs, "missed": missed_idcs, "extra": extra_idcs, "longshort": longshort_idcs}

    subspaces = {"rr": rr, "drrs": drrs, "mrrs": mrrs, "s12": s12, "s22": s22, "c1": c1, "c2": c2}

    return artifacts, subspaces


def _compute_threshold(signal, alpha, window_width):

    df = pd.DataFrame({"signal": np.abs(signal)})
    q1 = df.rolling(window_width, center=True, min_periods=1).quantile(0.25).signal.to_numpy()
    q3 = df.rolling(window_width, center=True, min_periods=1).quantile(0.75).signal.to_numpy()
    th = alpha * ((q3 - q1) / 2)

    return th


def _correct_artifacts(artifacts, peaks):

    # Artifact correction
    #####################
    # The integrity of indices must be maintained if peaks are inserted or
    # deleted: for each deleted beat, decrease indices following that beat in
    # all other index lists by 1. Likewise, for each added beat, increment the
    # indices following that beat in all other lists by 1.
    extra_idcs = artifacts["extra"]
    missed_idcs = artifacts["missed"]
    ectopic_idcs = artifacts["ectopic"]
    longshort_idcs = artifacts["longshort"]

    # Delete extra peaks.
    if extra_idcs:
        peaks = _correct_extra(extra_idcs, peaks)
        # Update remaining indices.
        missed_idcs = _update_indices(extra_idcs, missed_idcs, -1)
        ectopic_idcs = _update_indices(extra_idcs, ectopic_idcs, -1)
        longshort_idcs = _update_indices(extra_idcs, longshort_idcs, -1)

    # Add missing peaks.
    if missed_idcs:
        peaks = _correct_missed(missed_idcs, peaks)
        # Update remaining indices.
        ectopic_idcs = _update_indices(missed_idcs, ectopic_idcs, 1)
        longshort_idcs = _update_indices(missed_idcs, longshort_idcs, 1)

    if ectopic_idcs:
        peaks = _correct_misaligned(ectopic_idcs, peaks)

    if longshort_idcs:
        peaks = _correct_misaligned(longshort_idcs, peaks)

    return peaks


def _correct_extra(extra_idcs, peaks):

    corrected_peaks = peaks.copy()
    corrected_peaks = np.delete(corrected_peaks, extra_idcs)

    return corrected_peaks


def _correct_missed(missed_idcs, peaks):

    corrected_peaks = peaks.copy()
    missed_idcs = np.array(missed_idcs)
    # Calculate the position(s) of new beat(s). Make sure to not generate
    # negative indices. prev_peaks and next_peaks must have the same
    # number of elements.
    valid_idcs = np.logical_and(missed_idcs > 1, missed_idcs < len(corrected_peaks))  # pylint: disable=E1111
    missed_idcs = missed_idcs[valid_idcs]
    prev_peaks = corrected_peaks[[i - 1 for i in missed_idcs]]
    next_peaks = corrected_peaks[missed_idcs]
    added_peaks = prev_peaks + (next_peaks - prev_peaks) / 2
    # Add the new peaks before the missed indices (see numpy docs).
    corrected_peaks = np.insert(corrected_peaks, missed_idcs, added_peaks)

    return corrected_peaks


def _correct_misaligned(misaligned_idcs, peaks):

    corrected_peaks = peaks.copy()
    misaligned_idcs = np.array(misaligned_idcs)
    # Make sure to not generate negative indices, or indices that exceed
    # the total number of peaks. prev_peaks and next_peaks must have the
    # same number of elements.
    valid_idcs = np.logical_and(
        misaligned_idcs > 1, misaligned_idcs < len(corrected_peaks) - 1  # pylint: disable=E1111
    )
    misaligned_idcs = misaligned_idcs[valid_idcs]
    prev_peaks = corrected_peaks[[i - 1 for i in misaligned_idcs]]
    next_peaks = corrected_peaks[[i + 1 for i in misaligned_idcs]]

    half_ibi = (next_peaks - prev_peaks) / 2
    peaks_interp = prev_peaks + half_ibi
    # Shift the R-peaks from the old to the new position.
    corrected_peaks = np.delete(corrected_peaks, misaligned_idcs)
    corrected_peaks = np.concatenate((corrected_peaks, peaks_interp)).astype(int)
    corrected_peaks.sort(kind="mergesort")

    return corrected_peaks


def _update_indices(source_idcs, update_idcs, update):
    """For every element s in source_idcs, change every element u in update_idcs according to update, if u is larger
    than s."""
    if not update_idcs:
        return update_idcs

    for s in source_idcs:
        update_idcs = [u + update if u > s else u for u in update_idcs]

    return update_idcs


def _plot_artifacts_lipponen2019(artifacts, info):

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
    gs = matplotlib.gridspec.GridSpec(ncols=4, nrows=3, width_ratios=[1, 2, 2, 2])
    fig = plt.figure(constrained_layout=False, figsize=(15, 10))
    ax0 = fig.add_subplot(gs[0, :-2])
    ax1 = fig.add_subplot(gs[1, :-2])
    ax2 = fig.add_subplot(gs[2, :-2])
    ax3 = fig.add_subplot(gs[:, -1])
    ax4 = fig.add_subplot(gs[:, -2])

    ax0.set_title("Artifact types", fontweight="bold")
    ax0.plot(rr, label="heart period")
    ax0.scatter(longshort_idcs, rr[longshort_idcs], marker="x", c="m", s=100, zorder=3, label="long/short")
    ax0.scatter(ectopic_idcs, rr[ectopic_idcs], marker="x", c="g", s=100, zorder=3, label="ectopic")
    ax0.scatter(extra_idcs, rr[extra_idcs], marker="x", c="y", s=100, zorder=3, label="false positive")
    ax0.scatter(missed_idcs, rr[missed_idcs], marker="x", c="r", s=100, zorder=3, label="false negative")
    ax0.legend(loc="upper right")

    # Visualize first threshold.
    ax1.set_title("Consecutive-difference criterion", fontweight="bold")
    ax1.plot(np.abs(drrs), label="normalized difference consecutive heart periods")
    ax1.axhline(1, c="r", label="artifact threshold")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 5)

    # Visualize second threshold.
    ax2.set_title("Difference-from-median criterion", fontweight="bold")
    ax2.plot(np.abs(mrrs), label="difference from median over 11 periods")
    ax2.axhline(3, c="r", label="artifact threshold")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, 5)

    # Visualize subspaces.
    ax4.set_title("Subspace 1", fontweight="bold")
    ax4.set_xlabel("S11")
    ax4.set_ylabel("S12")
    ax4.scatter(drrs, s12, marker="x", label="heart periods")
    ax4.set_ylim(-5, 5)
    ax4.set_xlim(-10, 10)
    verts0 = [(-10, 5), (-10, -c1 * -10 + c2), (-1, -c1 * -1 + c2), (-1, 5)]

    poly0 = matplotlib.patches.Polygon(verts0, alpha=0.3, facecolor="r", edgecolor=None, label="ectopic periods")
    ax4.add_patch(poly0)
    verts1 = [(1, -c1 * 1 - c2), (1, -5), (10, -5), (10, -c1 * 10 - c2)]
    poly1 = matplotlib.patches.Polygon(verts1, alpha=0.3, facecolor="r", edgecolor=None)
    ax4.add_patch(poly1)
    ax4.legend(loc="upper right")

    ax3.set_title("Subspace 2", fontweight="bold")
    ax3.set_xlabel("S21")
    ax3.set_ylabel("S22")
    ax3.scatter(drrs, s22, marker="x", label="heart periods")
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-10, 10)
    verts2 = [(-10, 10), (-10, 1), (-1, 1), (-1, 10)]
    poly2 = matplotlib.patches.Polygon(verts2, alpha=0.3, facecolor="r", edgecolor=None, label="short periods")
    ax3.add_patch(poly2)
    verts3 = [(1, -1), (1, -10), (10, -10), (10, -1)]
    poly3 = matplotlib.patches.Polygon(verts3, alpha=0.3, facecolor="y", edgecolor=None, label="long periods")
    ax3.add_patch(poly3)
    ax3.legend(loc="upper right")


# =============================================================================
# Neurokit
# =============================================================================
def _remove_small(peaks, sampling_rate=1000, interval_min=None, relative_interval_min=None, robust=False):
    if interval_min is None and relative_interval_min is None:
        return peaks

    if interval_min is not None:
        interval = signal_period(peaks, sampling_rate=sampling_rate, desired_length=None)
        peaks = peaks[interval > interval_min]

    if relative_interval_min is not None:
        interval = signal_period(peaks, sampling_rate=sampling_rate, desired_length=None)
        peaks = peaks[standardize(interval, robust=robust) > relative_interval_min]

    return peaks


def _interpolate_big(peaks, sampling_rate=1000, interval_max=None, relative_interval_max=None, robust=False):
    if interval_max is None and relative_interval_max is None:
        return peaks

    continue_loop = True
    while continue_loop is True:
        if interval_max is not None:
            interval = signal_period(peaks, sampling_rate=sampling_rate, desired_length=None)
            peaks, continue_loop = _interpolate_missing(peaks, interval, interval_max, sampling_rate)

        if relative_interval_max is not None:
            interval = signal_period(peaks, sampling_rate=sampling_rate, desired_length=None)
            interval = standardize(interval, robust=robust)
            peaks, continue_loop = _interpolate_missing(peaks, interval, interval_max, sampling_rate)

    return peaks


def _interpolate_missing(peaks, interval, interval_max, sampling_rate):
    outliers = interval > interval_max
    outliers_loc = np.where(outliers)[0]
    if np.sum(outliers) == 0:
        return peaks, False

    # Delete large interval and replace by two unknown intervals
    interval[outliers] = np.nan
    interval = np.insert(interval, outliers_loc, np.nan)
    #    new_peaks_location = np.where(np.isnan(interval))[0]

    # Interpolate values
    interval = pd.Series(interval).interpolate().values
    peaks_corrected = _period_to_location(interval, sampling_rate, first_location=peaks[0])
    peaks = np.insert(peaks, outliers_loc, peaks_corrected[outliers_loc + np.arange(len(outliers_loc))])
    return peaks, True


def _period_to_location(period, sampling_rate=1000, first_location=0):
    location = np.cumsum(period * sampling_rate)
    location = location - (location[0] - first_location)
    return location.astype(np.int)

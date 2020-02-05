# -*- coding: utf-8 -*-
import numpy as np

from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize
from ..signal import signal_resample



def ecg_rate(rpeaks, sampling_rate=1000, desired_length=None, artifacts=None):
    """Calculate heart rate from R-peaks.

    Parameters
    ----------
    rpeaks : dict
        The samples at which the R-peak occur. Dict returned by
        `ecg_findpeaks()`.
    sampling_rate : int
        The sampling frequency of the signal that contains the R-peaks (in Hz,
        i.e., samples/second). Defaults to 1000.
    desired_length : int
        By default, the returned heart rate has the same number of elements as
        peaks. If set to an integer, the returned heart rate will be
        interpolated between R-peaks over `desired_length` samples. Has no
        effect if a DataFrame is passed in as the `peaks` argument. Defaults to
        None.
    artifacts : dict
        Dictionary containing indices of erroneous inter-beat-intervals,
        obtained from either `ecg_fixpeaks()` or `ecg_process()`. Default is
        None. If a dict is provided, heart rate will be corrected according to
        Jukka A. Lipponen & Mika P. Tarvainen (2019): A robust algorithm for
        heart rate variability time series artefact correction using novel beat
        classification, Journal of Medical Engineering & Technology,
        DOI: 10.1080/03091902.2019.1640306.

    Returns
    -------
    array
        A Numpy array containing the heart rate.

    See Also
    --------
    ecg_clean, ecg_findpeaks, ecg_fixpeaks, ecg_process, ecg_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>> import matplotlib.pyplot as plt
    >>> ecg = nk.ecg_simulate(duration=15, sampling_rate=1000, heart_rate=80)
    >>> rpeaks = nk.ecg_findpeaks(ecg)
    >>> artifacts = nk.ecg_fixpeaks(rpeaks, show=True)
    >>> rate_corrected = nk.ecg_rate(rpeaks, artifacts=artifacts,
    >>>                              desired_length=len(ecg))
    >>> rate_uncorrected = nk.ecg_rate(rpeaks, desired_length=len(ecg))
    >>>
    >>> fig, ax = plt.subplots()
    >>> ax.plot(rate_uncorrected, label="heart rate without artifact correction")
    >>> ax.plot(rate_corrected, label="heart rate with artifact correction")
    >>> ax.legend(loc="upper right")
    """
    # Get R-peaks indices from DataFrame or dict.
    rpeaks, _ = _signal_formatpeaks_sanitize(rpeaks, desired_length=None)

    # Sanity check artifacts.
    if isinstance(artifacts, dict):
        if not np.any(list(artifacts.values())):
            # If none of the artifact types contains any indices, skip artifact
            # correction-
            artifacts = None

    if isinstance(artifacts, dict):

        extra_idcs = artifacts["extra"]
        missed_idcs = artifacts["missed"]
        ectopic_idcs = artifacts["ectopic"]
        longshort_idcs = artifacts["longshort"]

        # Delete extra peaks.
        if extra_idcs:
            rpeaks = np.delete(rpeaks, extra_idcs)
            # Re-calculate the RR series.
            rr = np.ediff1d(rpeaks, to_begin=0) / sampling_rate
    #        print('extra: {}'.format(peaks[extra_idcs]))
            # Update remaining indices.
            missed_idcs = update_indices(extra_idcs, missed_idcs, -1)
            ectopic_idcs = update_indices(extra_idcs, ectopic_idcs, -1)
            longshort_idcs = update_indices(extra_idcs, longshort_idcs, -1)

        # Add missing peaks.
        if missed_idcs:
            # Calculate the position(s) of new beat(s).
            prev_peaks = rpeaks[[i - 1 for i in missed_idcs]]
            next_peaks = rpeaks[missed_idcs]
            added_peaks = prev_peaks + (next_peaks - prev_peaks) / 2
            # Add the new peaks.
            rpeaks = np.insert(rpeaks, missed_idcs, added_peaks)
            # Re-calculate the RR series.
            rr = np.ediff1d(rpeaks, to_begin=0) / sampling_rate
    #        print('missed: {}'.format(peaks[missed_idcs]))
            # Update remaining indices.
            ectopic_idcs = update_indices(missed_idcs, ectopic_idcs, 1)
            longshort_idcs = update_indices(missed_idcs, longshort_idcs, 1)

        # Interpolate ectopic as well as long or short peaks (important to do
        # this after peaks are deleted and/or added).
        interp_idcs = np.concatenate((ectopic_idcs, longshort_idcs)).astype(int)
        if interp_idcs.size > 0:
            interp_idcs.sort(kind='mergesort')
            # Ignore the artifacts during interpolation
            x = np.delete(np.arange(0, rr.size), interp_idcs)
            # Interpolate artifacts
            interp_artifacts = np.interp(interp_idcs, x, rr[x])
            rr[interp_idcs] = interp_artifacts
    #        print('interpolated: {}'.format(peaks[interp_idcs]))

    elif artifacts is None:

        rr = np.ediff1d(rpeaks, to_begin=0) / sampling_rate

    # The rate corresponding to the first peak is set to the mean RR.
    rr[0] = np.mean(rr)
    rate = 60 / rr

    if desired_length:
        rate = signal_resample(rate, desired_length=desired_length,
                               sampling_rate=sampling_rate)

    return rate


def update_indices(source_idcs, update_idcs, update):
    """
    for every element s in source_idcs, change every element u in update_idcs
    accoridng to update, if u is larger than s
    """
    update_idcs_buffer = update_idcs
    for s in source_idcs:
        # for each list, find the indices (of indices) that need to be updated
        updates = [i for i, j in enumerate(update_idcs) if j > s]
#        print('updates: {}'.format(updates))
        for u in updates:
            update_idcs_buffer[u] += update
#        print('update_idcs: {}'.format(update_idcs_buffer))
    return update_idcs_buffer

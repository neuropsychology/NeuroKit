# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..events import events_find
from ..misc import as_vector
from ..signal import signal_binarize, signal_changepoints, signal_formatpeaks


def emg_activation(
    emg_amplitude=None,
    emg_cleaned=None,
    sampling_rate=1000,
    method="threshold",
    threshold="default",
    duration_min="default",
    **kwargs
):
    """
    Detects onset in EMG signal based on the amplitude threshold.

    Parameters
    ----------
    emg_amplitude, emg_cleaned : array
        At least one EMG-related signal. Either the amplitude of the EMG signal, obtained from ``emg_amplitude()`` (for methods like 'threshold' or 'mixture'), and / or the cleaned EMG signal (for methods like 'pelt').
    sampling_rate : int
        The sampling frequency of ``emg_signal`` (in Hz, i.e., samples/second).
     method : str
        The algorithm used to discriminate between activity and baseline. Can be one of 'mixture'
        (default) or 'threshold'. If 'mixture', will use a Gaussian Mixture Model to categorize
        between the two states. If 'threshold', will consider as activated all points which
        amplitude is superior to the threshold.
    threshold : float
        If ``method`` is 'mixture', then it corresponds to the minimum probability required
        to be considered as activated (default to 0.33). If `method` is 'threshold', then
        it corresponds to the minimum amplitude to detect as onset. Defaults to one
        tenth of the standard deviation of ``emg_amplitude``.
    duration_min : float
        The minimum duration of a period of activity or non-activity in seconds.
        If 'default', will be set to 0.05 (50 ms).

    Returns
    -------
    info : dict
        A dictionary containing additional information,
        in this case the samples at which the onsets, offsets, and periods of
        activations of the EMG signal occur, accessible with the
        key "EMG_Onsets", "EMG_Offsets", and "EMG_Activity" respectively.
    activity_signal : DataFrame
        A DataFrame of same length as the input signal in which occurences of
        onsets, offsets, and activity (above the threshold) of the EMG signal
        are marked as "1" in lists of zeros with the same length as
        `emg_amplitude`. Accessible with the keys "EMG_Onsets",
        "EMG_Offsets", and "EMG_Activity" respectively.

    See Also
    --------
    emg_simulate, emg_clean, emg_amplitude, emg_process, emg_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Simulate signal and obtain amplitude
    >>> emg = nk.emg_simulate(duration=10, burst_number=3)
    >>> emg_cleaned = nk.emg_clean(emg)
    >>> emg_amplitude = nk.emg_amplitude(emg_cleaned)
    >>>
    >>> # Threshold method
    >>> activity, info = nk.emg_activation(emg_amplitude=emg_amplitude, method="threshold")
    >>> fig = nk.events_plot([info["EMG_Offsets"], info["EMG_Onsets"]], emg_cleaned)
    >>> fig #doctest: +SKIP
    >>>
    >>> # Threshold method
    >>> activity, info = nk.emg_activation(emg_cleaned=emg_cleaned, method="pelt")
    >>> nk.signal_plot([emg_cleaned, activity["EMG_Activity"]])
    >>> fig #doctest: +SKIP

    References
    ----------
    - BioSPPy: https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/emg.py

    """
    # Sanity checks.
    if emg_amplitude is not None:
        emg_amplitude = as_vector(emg_amplitude)
    if emg_cleaned is not None:
        emg_cleaned = as_vector(emg_cleaned)
        if emg_amplitude is None:
            emg_amplitude = as_vector(emg_cleaned)

    if duration_min == "default":
        duration_min = int(0.05 * sampling_rate)

    # Find offsets and onsets.
    method = method.lower()  # remove capitalised letters
    if method == "threshold":
        if emg_amplitude is None:
            raise ValueError(
                "NeuroKit error: emg_activation(): 'threshold' method needs 'emg_amplitude' signal to be passed."
            )
        activity = _emg_activation_threshold(emg_amplitude, threshold=threshold)
    elif method == "mixture":
        if emg_amplitude is None:
            raise ValueError(
                "NeuroKit error: emg_activation(): 'mixture' method needs 'emg_amplitude' signal to be passed."
            )
        activity = _emg_activation_mixture(emg_amplitude, threshold=threshold)
    elif method == "pelt":
        if emg_cleaned is None:
            raise ValueError(
                "NeuroKit error: emg_activation(): 'pelt' method needs 'emg_cleaned' (cleaned or raw EMG) signal to be passed."
            )
        activity = _emg_activation_pelt(emg_cleaned, duration_min=duration_min, **kwargs)
    else:
        raise ValueError(
            "NeuroKit error: emg_activation(): 'method' should be one of 'mixture', 'threshold', or 'pelt'."
        )

    # Sanitize activity.
    info = _emg_activation_activations(activity, sampling_rate=sampling_rate, duration_min=duration_min)

    # Prepare Output.
    df_activity = signal_formatpeaks(
        {"EMG_Activity": info["EMG_Activity"]}, desired_length=len(emg_amplitude), peak_indices=info["EMG_Activity"]
    )
    df_onsets = signal_formatpeaks(
        {"EMG_Onsets": info["EMG_Onsets"]}, desired_length=len(emg_amplitude), peak_indices=info["EMG_Onsets"]
    )
    df_offsets = signal_formatpeaks(
        {"EMG_Offsets": info["EMG_Offsets"]}, desired_length=len(emg_amplitude), peak_indices=info["EMG_Offsets"]
    )

    # Modify output produced by signal_formatpeaks.
    for x in range(len(emg_amplitude)):
        if df_activity["EMG_Activity"][x] != 0:
            if df_activity.index[x] == df_activity.index.get_loc(x):
                df_activity["EMG_Activity"][x] = 1
            else:
                df_activity["EMG_Activity"][x] = 0
        if df_offsets["EMG_Offsets"][x] != 0:
            if df_offsets.index[x] == df_offsets.index.get_loc(x):
                df_offsets["EMG_Offsets"][x] = 1
            else:
                df_offsets["EMG_Offsets"][x] = 0

    activity_signal = pd.concat([df_activity, df_onsets, df_offsets], axis=1)

    return activity_signal, info


# =============================================================================
# Methods
# =============================================================================


def _emg_activation_threshold(emg_amplitude, threshold="default"):

    if threshold == "default":
        threshold = (1 / 10) * np.std(emg_amplitude)

    if threshold > np.max(emg_amplitude):
        raise ValueError(
            "NeuroKit error: emg_activation(): the threshold specified exceeds the maximum of the signal" "amplitude."
        )

    activity = signal_binarize(emg_amplitude, method="threshold", threshold=threshold)
    return activity


def _emg_activation_mixture(emg_amplitude, threshold="default"):

    if threshold == "default":
        threshold = 0.33

    activity = signal_binarize(emg_amplitude, method="mixture", threshold=threshold)
    return activity


def _emg_activation_pelt(emg_cleaned, threshold="default", duration_min=0.05, **kwargs):

    if threshold == "default":
        threshold = None

    # Get changepoints
    changepoints = signal_changepoints(emg_cleaned, change="var", show=False, **kwargs)

    # Add first point
    if changepoints[0] != 0:
        changepoints = np.append(0, changepoints)

    # Sanitize
    lengths = np.append(0, np.diff(changepoints))
    changepoints = changepoints[1:][lengths[1:] > duration_min]

    # reèAdd first point
    if changepoints[0] != 0:
        changepoints = np.append(0, changepoints)

    binary = np.full(len(emg_cleaned), np.nan)
    binary[changepoints[0::2]] = 0
    binary[changepoints[1::2]] = 1

    activity = pd.Series(binary).fillna(method="ffill").values

    # Label as 1 to parts that have the larger SD (likely to be activations)
    if emg_cleaned[activity == 1].std() > emg_cleaned[activity == 0].std():
        activity = np.abs(activity - 1)

    activity[0] = 0
    activity[-1] = 0

    return activity


# =============================================================================
# Internals
# =============================================================================
def _emg_activation_activations(activity, sampling_rate=1000, duration_min=0.05):

    activations = events_find(activity, threshold=0.5, threshold_keep="above", duration_min=duration_min)
    activations["offset"] = activations["onset"] + activations["duration"]

    baseline = events_find(activity == 0, threshold=0.5, threshold_keep="above", duration_min=duration_min)
    baseline["offset"] = baseline["onset"] + baseline["duration"]

    # Cross-comparison
    valid = np.isin(activations["onset"], baseline["offset"])
    onsets = activations["onset"][valid]
    offsets = activations["offset"][valid]

    new_activity = np.array([])
    for x, y in zip(onsets, offsets):
        activated = np.arange(x, y)
        new_activity = np.append(new_activity, activated)

    # Prepare Output.
    info = {"EMG_Onsets": onsets, "EMG_Offsets": offsets, "EMG_Activity": new_activity}

    return info

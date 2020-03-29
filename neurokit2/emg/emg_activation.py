# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..events import events_find
from ..signal import signal_formatpeaks
from ..signal import signal_binarize

def emg_activation(emg_amplitude, sampling_rate=1000, method="mixture", threshold='default', duration_min="default"):
    """Detects onset in EMG signal based on the amplitude threshold.

    Parameters
    ----------
    emg_amplitude : array
        The amplitude of the emg signal, obtained from `emg_amplitude()`.
    sampling_rate : int
        The sampling frequency of `emg_signal` (in Hz, i.e., samples/second).
     method : str
        The algorithm used to discriminate between activity and baseline. Can be one of 'mixture'
        (default) or 'threshold'. If 'mixture', will use a Gaussian Mixture Model to categorize
        between the two states. If 'threshold', will consider as activated all points which
        amplitude is superior to the threshold.
    threshold : float
        If `method` is 'mixture', then it corresponds to the minimum probability required
        to be considered as activated (default to 0.5). If `method` is 'threshold', then
        it corresponds to the minimum amplitude to detect as onset. Defaults to one
        tenth of the standard deviation of `emg_amplitude`.
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
    >>> cleaned = nk.emg_clean(emg)
    >>> emg_amplitude = nk.emg_amplitude(cleaned)
    >>>
    >>> activity_signal,info = nk.emg_activation(emg_amplitude)
    >>> nk.events_plot([info["EMG_Offsets"], info["EMG_Onsets"]],
                       emg_amplitude)

    References
    ----------
    - BioSPPy: https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/emg.py
    """
    # Sanity checks.
    if not isinstance(emg_amplitude, np.ndarray):
        emg_amplitude = np.atleast_1d(emg_amplitude).astype('float64')

    # Find offsets and onsets.
    method = method.lower()  # remove capitalised letters
    if method == "threshold":
        activity = _emg_activation_threshold(emg_amplitude, threshold=threshold)
    elif method == "mixture":
        activity = _emg_activation_mixture(emg_amplitude, threshold=threshold)
    else:
        raise ValueError("NeuroKit error: emg_activation(): 'method' should be "
                         "one of 'mixture' or 'threshold'.")

    # Sanitize activity.
    info = _emg_activation_activations(activity, sampling_rate=sampling_rate, duration_min=duration_min)


    # Prepare Output.
    df_activity = signal_formatpeaks({"EMG_Activity": info["EMG_Activity"]},
                                     desired_length=len(emg_amplitude),
                                     peak_indices=info["EMG_Activity"])
    df_onsets = signal_formatpeaks({"EMG_Onsets": info["EMG_Onsets"]},
                                   desired_length=len(emg_amplitude),
                                   peak_indices=info["EMG_Onsets"])
    df_offsets = signal_formatpeaks({"EMG_Offsets": info["EMG_Offsets"]},
                                    desired_length=len(emg_amplitude),
                                    peak_indices=info["EMG_Offsets"])

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

def _emg_activation_threshold(emg_amplitude, threshold='default'):

    if threshold == 'default':
        threshold = (1/10)*np.std(emg_amplitude)

    if threshold > np.max(emg_amplitude):
        raise ValueError("NeuroKit error: emg_activation(): threshold"
                         "specified exceeds the maximum of the signal"
                         "amplitude.")

    activity = signal_binarize(emg_amplitude, method="threshold", threshold=threshold)
    return activity



def _emg_activation_mixture(emg_amplitude, threshold="default"):

    if threshold == 'default':
        threshold = 0.5

    activity = signal_binarize(emg_amplitude, method="mixture", threshold=threshold)
    return activity



#def _emg_activation_powerbased(emg_cleaned, sampling_rate=1000, threshold=0.75):
#    """
#    >>> emg_cleaned = nk.emg_simulate(duration=20, n_bursts=3)
#    >>> binarized_energy, info = _emg_activation_powerbased(emg_cleaned)
#    >>> nk.signal_plot([emg_cleaned, binarized_energy], standardize=True)
#    >>> nk.events_plot(info["EMG_Onsets"], emg_cleaned)
#    >>> nk.events_plot(info["EMG_Onsets"], energy.values)
#    """
#
#    # Preprocessing
#    signal = np.abs(emg_cleaned)  # getting absolute value of the EMG channel
#
#    # Parameters
#    window = np.int(0.5 * sampling_rate)  # related to the duration of a movement
##    step = 100  # equivalent to resolution
#
#    # rolling window
#    energy = pd.Series(signal).rolling(window, win_type='boxcar').sum()  # [::step]
#
#    # Get onsets
#    binarized_energy = nk.signal_binarize(energy.fillna(0), threshold=energy.quantile(threshold))
#    activations = nk.events_find(binarized_energy, duration_min=window)
#
#    # Prepare Output.
#    info = {"EMG_Onsets": activations["onset"],
#            "EMG_Offsets": activations["onset"] + activations["duration"]}
#
#    return binarized_energy, info




# =============================================================================
# Internals
# =============================================================================
def _emg_activation_activations(activity, sampling_rate=1000, duration_min="default"):

    if duration_min == "default":
        duration_min = int(0.05 * sampling_rate)

    activations = events_find(activity, threshold=0.5, threshold_keep='above', duration_min=duration_min)
    activations["offset"] = activations["onset"] + activations["duration"]

    baseline = events_find(activity == 0, threshold=0.5, threshold_keep='above', duration_min=duration_min)
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
    info = {"EMG_Onsets": onsets,
            "EMG_Offsets": offsets,
            "EMG_Activity": new_activity}

    return info


#def _emg_activation_offsets_onsets(activity):
#
#    # Extract indices of activated data points.
#    activated = np.nonzero(activity)[0]
#    baseline = np.nonzero(activity == False)[0]
#
#    onsets = np.intersect1d(activated - 1, baseline)
#    offsets = np.intersect1d(activated + 1, baseline)
#
#    # Check that indices do not include first and last sample point.
#    for i in zip(onsets, offsets):
#        if i == 0 or i == len(activity-1):
#            onsets.remove(i)
#            offsets.remove(i)
#
#    # Extract indexes of activated samples
#    activations = np.array([])
#    for x, y in zip(onsets, offsets):
#        activated = np.arange(x, y)
#        activations = np.append(activations, activated)
#
#    # Prepare Output.
#    info = {"EMG_Onsets": onsets,
#            "EMG_Offsets": offsets,
#            "EMG_Activity": activations}
#    return info

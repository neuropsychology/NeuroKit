# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_formatpeaks


def emg_activation(emg_amplitude, threshold='default'):
    """Detects onset in EMG signal based on the amplitude threshold.

    Parameters
    ----------
    emg_amplitude : array
        The amplitude of the emg signal, obtained from `emg_amplitude()`.
    threshold : float
        The minimum amplitude to detect as onset. Defaults to one tenth of the
        standard deviation of `emg_amplitude`.

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
    >>> emg = nk.emg_simulate(duration=10, sampling_rate=250, n_bursts=3)
    >>> cleaned = nk.emg_clean(emg, sampling_rate=250)
    >>> emg_amplitude = nk.emg_amplitude(cleaned)
    >>>
    >>> activity_signal,info = nk.emg_activation(emg_amplitude, threshold)
    >>> nk.events_plot([info["EMG_Offsets"], info["EMG_Onsets"]],
                       emg_amplitude)

    References
    ----------
    - BioSPPy: https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/emg.py
    """
    if threshold == 'default':
        threshold = (1/10)*np.std(emg_amplitude)
    else:
        threshold = threshold

    # Sanity checks.
    if not isinstance(emg_amplitude, np.ndarray):
        emg_amplitude = np.atleast_1d(emg_amplitude).astype('float64')
    if threshold > np.max(emg_amplitude):
        raise ValueError("NeuroKit error: emg_activation(): threshold"
                         "specified exceeds the maximum of the signal"
                         "amplitude.")

    # Extract indices of data points greater than or equal to threshold.
    above = np.nonzero(emg_amplitude >= threshold)[0]
    below = np.nonzero(emg_amplitude < threshold)[0]

    onsets = np.intersect1d(above - 1, below)
    offsets = np.intersect1d(above + 1, below)

    # Check that indices do not include first and last sample point.
    for i in zip(onsets, offsets):
        if i == 0 or i == len(emg_amplitude-1):
            onsets.remove(i)
            offsets.remove(i)

    # Extract indexes of activated samples
    activations = np.array([])
    for x, y in zip(onsets, offsets):
        activated = np.arange(x, y)
        activations = np.append(activations, activated)

    # Prepare Output.
    info = {"EMG_Onsets": onsets,
            "EMG_Offsets": offsets,
            "EMG_Activity": activations}
    info_activity = {"EMG_Activity": activations}
    info_onsets = {"EMG_Onsets": onsets}
    info_offsets = {"EMG_Offsets": offsets}

    df_activity = signal_formatpeaks(info_activity,
                                     desired_length=len(emg_amplitude),
                                     peak_indices=info_activity["EMG_Activity"])
    df_onsets = signal_formatpeaks(info_onsets,
                                   desired_length=len(emg_amplitude),
                                   peak_indices=info_onsets["EMG_Onsets"])
    df_offsets = signal_formatpeaks(info_offsets,
                                    desired_length=len(emg_amplitude),
                                    peak_indices=info_offsets["EMG_Offsets"])

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

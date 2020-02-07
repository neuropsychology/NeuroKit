# -*- coding: utf-8 -*-
import numpy as np

from ..signal import signal_formatpeaks


def emg_activation(emg_amplitude, threshold=0.01):
    """Detects onset in EMG signal based on the amplitude threshold.

    Parameters
    ----------
    emg_amplitude : array
        The amplitude of the emg signal, obtained from `emg_amplitude()`.
    threshold : float
        The minimum amplitude to detect as onset. Defaults to 0.01.

    Returns
    -------
    info : dict
        A dictionary containing additional information,
        in this case the samples at which the onsets of the amplitude occur,
        accessible with the key "EMG_Onsets".
    activity_signal : DataFrame
        A DataFrame of same length as the input signal in which occurences of
        EMG activity (above the threshold) are marked as "1" in
        lists of zeros with the same length as `emg_amplitude`.
        Accessible with the keys "EMG_Activity".

    See Also
    --------
    emg_simulate, emg_clean, emg_amplitude, emg_process, emg_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Simulate signal and obtain amplitude
    >>> emg = nk.emg_simulate(duration=10, sampling_rate=1000, n_bursts=3)
    >>> cleaned = nk.emg_clean(emg, sampling_rate=1000)
    >>> emg_amplitude = nk.emg_amplitude(cleaned)
    >>>
    >>> _,info = nk.emg_activation(emg_amplitude, threshold=0.1)
    >>> nk.events_plot([info["EMG_Offsets"], info["EMG_Onsets"]],
                       emg_amplitude)

    References
    ----------
    - BioSPPy: https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/emg.py
    """
    # Sanity checks.
    if not isinstance(emg_amplitude, np.ndarray):
        emg_amplitude = np.atleast_1d(emg_amplitude).astype('float64')
    if threshold > np.max(emg_amplitude):
        raise ValueError("NeuroKit error: emg_onsets(): threshold"
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
    info_emg_activity = {"EMG_Activity": activations}
    activity_signal = signal_formatpeaks(info_emg_activity,
                                         desired_length=len(emg_amplitude),
                                         peak_indices=info_emg_activity["EMG_Activity"])

    return activity_signal, info

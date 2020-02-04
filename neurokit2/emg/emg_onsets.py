# -*- coding: utf-8 -*-
import numpy as np


def emg_onsets(emg_amplitude, threshold=0, threshold2=None):
    """Detects onset in EMG signal based on the amplitude threshold,
    described by Marcos Duarte at <https://nbviewer.jupyter.org/github/
    demotu/BMC/blob/master/notebooks/DetectOnset.ipynb>

    Parameters
    ----------
    emg_amplitude : array
        The amplitude of the emg signal, obtained from `emg_amplitude()`.
    threshold, threshold2 : float
        The minimum amplitude to detect as onset.
        First threshold parameter defaults to 0 and second
        threshold parameter defaults to None.
        Second threshold parameter can be specified to avoid
        detecting baseline fluctuations i.e.,
        spurious signals that are above the first threshold.

    Returns
    -------
    info : dict
        A dictionary containing additional information,
        in this case the samples at which the onsets of the amplitude occur,
        accessible with the key "EMG_Onsets".

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
    >>> info = nk.emg_onsets(emg_amplitude, threshold=0.1)
    >>> nk.events_plot(info["EMG_Onsets"], emg_amplitude)

    References
    ----------
    - BMCLab: http://nbviewer.ipython.org/github/demotu/
    BMC/blob/master/notebooks/DetectOnset.ipynb
    """
    # Sanity checks.
    if not isinstance(emg_amplitude, np.ndarray):
        emg_amplitude = np.atleast_1d(emg_amplitude).astype('float64')
    if threshold > np.max(emg_amplitude):
        raise ValueError("NeuroKit error: emg_onsets(): threshold"
                         "specified exceeds the maximum of the signal"
                         "amplitude.")
    if threshold2 is not None and threshold2 > np.max(emg_amplitude):
        raise ValueError("NeuroKit error: emg_onsets(): threshold2"
                         "specified exceeds the maximum of the signal"
                         "amplitude.")

    # Extract indices of data points greater than or equal to threshold.
    indices = np.nonzero(emg_amplitude >= threshold)[0]

    # Extract initial and final indexes of each activity burst.
    indices = np.vstack((indices[np.diff(np.hstack((-np.inf, indices))) > 1],
                         indices[np.diff(np.hstack((indices, np.inf))) > 1])).T
    indices = indices[indices[:, 1]-indices[:, 0] >= 0, :]

    # Threshold2.
    if threshold2 is not None:
        indices2 = np.ones(indices.shape[0], dtype=bool)
        for i in range(indices.shape[0]):
            if np.count_nonzero(emg_amplitude[indices[i, 0]: indices[i, 1]+1] >= threshold2) < 1:
                indices2[i] = False
        indices = indices[indices2, :]

    # Prepare output.
    indices = list(np.concatenate(indices))
    info = {"EMG_Onsets": indices}

    return info

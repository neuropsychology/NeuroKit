# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..signal import signal_resample


def emg_simulate(duration=10, length=None, sampling_rate=1000, noise=0.01,
                 n_bursts=1, duration_bursts=1.0, random_state=42):
    """Simulate an EMG signal

    Generate an artificial (synthetic) EMG signal of a given duration and sampling rate.

    Parameters
    ----------
    duration : int
        Desired recording length in seconds.
    sampling_rate, length : int
        The desired sampling rate (in Hz, i.e., samples/second) or the desired
        length of the signal (in samples).
    noise : float
        Noise level (gaussian noise).
    n_bursts : int
        Desired number of bursts of activity (active muscle periods).
    duration_bursts : float or list
        Duration of the bursts. Can be a float (each burst will have the same
        duration) or a list of durations for each bursts.
    random_state: int
        Seed for the random number generator.

    Returns
    ----------
    array
        Vector containing the EMG signal.

    Examples
    ----------
    >>> import neurokit as nk
    >>> import pandas as pd
    >>>
    >>> emg = nk.emg_simulate(duration=10, n_bursts=3)
    >>> nk.signal_plot(emg)

    See Also
    --------
    ecg_simulate, rsp_simulate, eda_simulate, ppg_simulate


    References
    -----------
    This function is based on `this script <https://scientificallysound.org/2016/08/11/python-analysing-emg-signals-part-1/>`_.
    """
    # seed the random generator for reproducible results
    np.random.seed(random_state)

    # Sanity checks
    if isinstance(duration_bursts, int) or isinstance(duration_bursts, float):
        duration_bursts = np.repeat(duration_bursts, n_bursts)

    if len(duration_bursts) > n_bursts:
        raise ValueError("NeuroKit error: emg_simulate(): 'duration_bursts'" \
                         " cannot be longer than the value of 'n_bursts'")

    total_duration_bursts = np.sum(duration_bursts)
    if total_duration_bursts > duration:
        raise ValueError("NeuroKit error: emg_simulate(): The total duration" \
                         " of bursts cannot exceed the total duration")

    # Generate bursts
    bursts = []
    for burst in range(n_bursts):
        bursts += [list(np.random.uniform(-1, 1, size=int(1000*duration_bursts[burst])) + 0.08)]

    # Generate quiet
    n_quiet = n_bursts + 1  # number of quiet periods (in between bursts)
    duration_quiet = (duration - total_duration_bursts) / n_quiet  # duration of each quiet period
    quiets = []
    for quiet in range(n_quiet):
        quiets += [list(np.random.uniform(-0.05, 0.05, size=int(1000*duration_quiet)) + 0.08)]

    # Merge the two
    emg = []
    for i in range(len(quiets)):
        emg += quiets[i]
        if i < len(bursts):
            emg += bursts[i]
    emg = np.array(emg)

    # Add random (gaussian distributed) noise
    emg += np.random.normal(0, noise, len(emg))

    # Resample
    emg = signal_resample(emg,
                          sampling_rate=1000,
                          desired_length=duration*sampling_rate,
                          desired_sampling_rate=sampling_rate)

    return(emg)

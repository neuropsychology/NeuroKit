# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy

from ..signal import signal_resample


def ecg_simulate(duration=10, length=None, sampling_rate=1000, noise=0.01,
                 heart_rate=60, random_state=42):
    """Simulate an ECG/EKG signal

    Generate an artificial (synthetic) ECG signal of a given duration and sampling rate. It uses a 'Daubechies' wavelet that roughly approximates a single cardiac cycle.

    Parameters
    ----------
    duration : int
        Desired recording length in seconds.
    sampling_rate, length : int
        The desired sampling rate (in Hz, i.e., samples/second) or the desired length of the signal (in samples).
    noise : float
        Noise level (gaussian noise).
    heart_rate : int
        Desired simulated heart rate (in beats per minute).
    random_state : int
        Seed for the random number generator.



    Returns
    ----------
    array
        Vector containing the ECG signal.

    Examples
    ----------
    >>> import neurokit as nk
    >>> import pandas as pd
    >>>
    >>> ecg = nk.ecg_simulate(duration=10, sampling_rate=100)
    >>> nk.signal_plot(ecg)

    See Also
    --------
    rsp_simulate, eda_simulate, ppg_simulate, emg_simulate


    References
    -----------
    This function is based on `this script <https://github.com/diarmaidocualain/ecg_simulation>`_.
    """

    # Seed the random generator for reproducible results
    np.random.seed(random_state)

    # The "Daubechies" wavelet is a rough approximation to a real, single, cardiac cycle
    cardiac = scipy.signal.wavelets.daub(10)

    # Add the gap after the pqrst when the heart is resting.
    cardiac = np.concatenate([cardiac, np.zeros(10)])

    # Caculate the number of beats in capture time period
    num_heart_beats = int(duration * heart_rate / 60)

    # Concatenate together the number of heart beats needed
    ecg = np.tile(cardiac , num_heart_beats)

    # Add random (gaussian distributed) noise
    ecg += np.random.normal(0, noise, len(ecg))

    # Resample
    ecg = signal_resample(ecg,
                          sampling_rate=int(len(ecg)/10),
                          desired_length=length,
                          desired_sampling_rate=sampling_rate)

    return(ecg)

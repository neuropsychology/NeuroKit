# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy

from ..signal import signal_resample


def ecg_simulate(duration=10, length=None, sampling_rate=1000, bpm=60, noise=0.01):
    """Simulate an ECG/EKG signal

    Generate an artificial ECG signal of a given duration and sampling rate (based on `this script <https://github.com/diarmaidocualain/ecg_simulation>`_).

    Parameters
    ----------
    duration : int
        Desired recording length in seconds.
    sampling_rate, length : int
        The desired sampling rate (in Hz, i.e., samples/second) or the desired length of the signal (in samples).
    bpm : int
        Desired simulated heart rate.
    noise : float
       Noise level.


    Returns
    ----------
   array
        Array containing the ECG signal.

    Example
    ----------
    >>> import neurokit as nk
    >>> import pandas as pd
    >>>
    >>> ecg = nk.ecg_simulate(duration=10, bpm=60, sampling_rate=1000, noise=0.01)
    >>> pd.Series(ecg).plot()

    See Also
    --------
    signal_resample
    """
    # The "Daubechies" wavelet is a rough approximation to a real, single, cardiac cycle
    cardiac = scipy.signal.wavelets.daub(10)
    # Add the gap after the pqrst when the heart is resting.
    cardiac = np.concatenate([cardiac, np.zeros(10)])

    # Caculate the number of beats in capture time period
    num_heart_beats = int(duration * bpm / 60)

    # Concatenate together the number of heart beats needed
    ecg = np.tile(cardiac , num_heart_beats)

    # Add random (gaussian distributed) noise
    noise = np.random.normal(0, noise, len(ecg))
    ecg = noise + ecg

    # Resample
    ecg = signal_resample(ecg, sampling_rate=1000, desired_length=length, desired_sampling_rate=sampling_rate)

    return(ecg)
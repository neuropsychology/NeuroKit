# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def rsp_simulate(duration=10, length=None, sampling_rate=1000, breathing_rate = 0.25, amplitude = 1):
    """Simulate a respiratory signal

    Generate an artificial (synthetic) respiratory signal of a given frequency, amplitude, and number of cycles. It uses a trigonometric sine wave that roughly approximates a single respiratory cycle.

    Parameters
    ----------
    duration : int
        Desired length of duration (s).
    sampling_rate, length : int
        The desired sampling rate (in Hz, i.e., samples/second) or the desired length of the signal (in samples).
    breathing_rate : float
        Desired number of cycles, or number of breaths, in one second (in Hz).
    amplitude : int
        The maximum displacement of each waveform, or breath, from the equilibrium or rest state (in millivolts).


    Returns
    ----------
    array
        Vector containing the respiratory signal.

    Examples
    ----------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration = 10, length = None, sampling_rate = 1000, breathing_rate = 0.5, amplitude = 2)
    >>> nk.signal_plot(rsp)

    See Also
    --------
    rsp_clean, rsp_findpeaks, rsp_rate, rsp_process, rsp_plot
"""
    # Generate number of samples automatically if length is unspecified
    if length is None:
        length = duration * sampling_rate

    # Generate 1000 values along the length of the duration
    t = np.linspace(0, duration, int(length))

    # Compute the value of sine computed by the following trigonometric function
    rsp = amplitude*np.sin(2*np.pi*breathing_rate*t)

    return(rsp)

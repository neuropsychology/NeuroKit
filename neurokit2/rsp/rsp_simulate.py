# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def rsp_simulate(duration=10, length=None, sampling_rate=1000, breathing_rate=1, amplitude=1):
    """Simulate a respiratory signal

    Generate an artificial (synthetic) respiratory signal of a given frequency, amplitude, and number of cycles. It uses a trigonometric sine wave that roughly approximates a single respiratory cycle.

    Parameters
    ----------
    cycles : int
        Desired total number of cycles, or number of breaths taken during the given duration.
    frequency : int
        Desired number of cycles in one second (in Hz).
    amplitude : int
        The maximum displacement of each waveform, or breath, from the equilibrium or rest state (in millivolts).

    Returns
    ----------
    Returns two arrays. The first array, 't', contains float values along the length of the duration. The second array, 'rsp', contains float values of the amplitude that exactly correspond to time 't'.

    Examples
    ----------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(cycles = 15, frequency = 1, amplitude = 1)
    >>> nk.signal_plot(rsp)

    See Also
    --------
    rsp_clean, rsp_findpeaks, rsp_rate, rsp_process, rsp_plot
"""
#    # Duration of one complete oscillation
#    time_cycle = 2*(np.pi/frequency)
#
#    # Duration of the entire respiratory signal (in seconds)
#    duration = time_cycle * cycles
#
#    if length is None:
#        length = duration * sampling_rate
#
#    # Generate 1000 values along the length of the duration
#    t = np.linspace(0, duration, int(length))
#
#    # Compute the value of sine computed by the following trigonometric function
#    rsp = amplitude*np.sin(frequency*t)
#
#    return(rsp)
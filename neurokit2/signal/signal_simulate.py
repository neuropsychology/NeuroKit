# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from ..misc import listify


def signal_simulate(duration=10, sampling_rate=1000, frequency=1,
                    amplitude=0.5):
    """Simulate a continuous signal.

    Parameters
    ----------
    duration : float
        Desired length of duration (s).
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    frequency : float or list
        Oscillatory frequency of the signal (in Hz, i.e., oscillations per
        second).
    amplitude : float or list
        Amplitude of the oscillations.

    Returns
    -------
    array
        The simulated signal.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> pd.DataFrame({"1Hz": nk.signal_simulate(duration=5, frequency=1),
                      "2Hz": nk.signal_simulate(duration=5, frequency=2),
                      "Multi": nk.signal_simulate(duration=5, frequency=[0.5, 3], amplitude=[0.5, 0.2])}).plot()
    """
    # Generate samples.
    seconds = np.linspace(0, duration, duration * sampling_rate)
    signal = np.zeros(seconds.size)
    params = listify(frequency=frequency, amplitude=amplitude)

    for i in range(len(params["frequency"])):
        signal += _signal_simulate_sinusoidal(x=seconds,
                                              frequency=params["frequency"][i],
                                              amplitude=params["amplitude"][i])

    return signal


# =============================================================================
# Simple Sinusoidal Model
# =============================================================================
def _signal_simulate_sinusoidal(x, frequency=100, amplitude=0.5):

    # Compute the value of sine computed by the following trigonometric
    # function
    signal = amplitude * np.sin(2 * np.pi * x * frequency)

    return signal

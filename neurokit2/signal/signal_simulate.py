# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..misc import listify





def signal_simulate(duration=10, length=None, sampling_rate=1000,
                    frequency=1, amplitude=0.5):
    """Simulate a continuous signal.


    Parameters
    ----------
    duration : float
        Desired length of duration (s).
    sampling_rate, length : int
        The desired sampling rate (in Hz, i.e., samples/second) or the desired
        length of the signal (in samples).
    frequency : float or list
        Oscillatory frequency of the signal (in Hz, i.e., oscillations per second).
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
    # Generate number of samples automatically if length is unspecified
    if length is None:
        length = duration * sampling_rate

    signal = np.zeros(int(length))

    params = listify(frequency=frequency, amplitude=amplitude)
    for i in range(len(params["frequency"])):
        signal += _signal_simulate_sinusoidal(duration=duration,
                                              length=length,
                                              sampling_rate=sampling_rate,
                                              frequency=params["frequency"][i],
                                              amplitude=params["amplitude"][i])

    return signal



# =============================================================================
# Simple Sinusoidal Model
# =============================================================================
def _signal_simulate_sinusoidal(duration=10, length=None, sampling_rate=1000,
                                frequency=100, amplitude=0.5):
    # Generate values along the length of the duration
    x = np.linspace(0, duration, int(length))

    # Compute the value of sine computed by the following trigonometric
    # function
    signal = amplitude*np.sin(2*np.pi*x*frequency)

    return signal

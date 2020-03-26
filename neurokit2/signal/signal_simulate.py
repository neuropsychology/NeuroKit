# -*- coding: utf-8 -*-
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

    n_samples = int(np.rint(duration * sampling_rate))
    period = 1 / sampling_rate
    seconds = np.arange(n_samples) * period

    signal = np.zeros(seconds.size)
    params = listify(frequency=frequency, amplitude=amplitude)

    for i in range(len(params["frequency"])):

        freq = params["frequency"][i]
        amp = params["amplitude"][i]
        # Apply a very conservative Nyquist criterion in order to ensure
        # sufficiently sampled signals.
        nyquist = sampling_rate * .1
        if freq > nyquist:
            raise ValueError(f"NeuroKit error: Please choose frequencies smaller than {nyquist}.")
        # Also make sure that at leat one period of the frequency can be
        # captured over the duration of the signal.
        if (1 / freq) > duration:
            raise ValueError(f"NeuroKit error: Please choose frequencies larger than {1 / duration}.")

        signal += _signal_simulate_sinusoidal(x=seconds,
                                              frequency=freq,
                                              amplitude=amp)

    return signal


# =============================================================================
# Simple Sinusoidal Model
# =============================================================================
def _signal_simulate_sinusoidal(x, frequency=100, amplitude=0.5):

    signal = amplitude * np.sin(2 * np.pi * frequency * x)

    return signal

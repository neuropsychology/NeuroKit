# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..misc import NeuroKitWarning, listify


def signal_simulate(duration=10, sampling_rate=1000, frequency=1, amplitude=0.5, noise=0, silent=False):
    """Simulate a continuous signal.

    Parameters
    ----------
    duration : float
        Desired length of duration (s).
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    frequency : float or list
        Oscillatory frequency of the signal (in Hz, i.e., oscillations per second).
    amplitude : float or list
        Amplitude of the oscillations.
    noise : float
        Noise level (amplitude of the laplace noise).
    silent : bool
        If False (default), might print warnings if impossible frequencies are queried.

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
    >>> fig = pd.DataFrame({"1Hz": nk.signal_simulate(duration=5, frequency=1),
    ...                     "2Hz": nk.signal_simulate(duration=5, frequency=2),
    ...                     "Multi": nk.signal_simulate(duration=5, frequency=[0.5, 3], amplitude=[0.5, 0.2])}).plot()
    >>> fig #doctest: +SKIP

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
        nyquist = sampling_rate * 0.1
        if freq > nyquist:
            if not silent:
                warn(
                    f"Skipping requested frequency"
                    f" of {freq} Hz since it cannot be resolved at the"
                    f" sampling rate of {sampling_rate} Hz. Please increase"
                    f" sampling rate to {freq * 10} Hz or choose frequencies"
                    f" smaller than or equal to {nyquist} Hz.",
                    category=NeuroKitWarning
                )
            continue
        # Also make sure that at leat one period of the frequency can be
        # captured over the duration of the signal.
        if (1 / freq) > duration:
            if not silent:
                warn(
                    f"Skipping requested frequency"
                    f" of {freq} Hz since its period of {1 / freq} seconds"
                    f" exceeds the signal duration of {duration} seconds."
                    f" Please choose frequencies larger than"
                    f" {1 / duration} Hz or increase the duration of the"
                    f" signal above {1 / freq} seconds.",
                    category=NeuroKitWarning
                )
            continue

        signal += _signal_simulate_sinusoidal(x=seconds, frequency=freq, amplitude=amp)
        # Add random noise
        if noise > 0:
            signal += np.random.laplace(0, noise, len(signal))

    return signal


# =============================================================================
# Simple Sinusoidal Model
# =============================================================================
def _signal_simulate_sinusoidal(x, frequency=100, amplitude=0.5):

    signal = amplitude * np.sin(2 * np.pi * frequency * x)

    return signal

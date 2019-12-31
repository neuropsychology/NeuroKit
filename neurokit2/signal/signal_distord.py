# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .signal_resample import signal_resample
from ..misc import listify


def signal_distord(signal, sampling_rate=1000, noise_amplitude=0.1, noise_frequency=100, noise_shape="laplace", powerline_amplitude=0, powerline_frequency=50):
    """Signal distortion.

    Add noise of a given frequency, amplitude and shape to a signal.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    noise_amplitude : float
        The amplitude of the noise (the scale of the random function, relative
        to the standard deviation of the signal).
    noise_frequency : int
        The frequency of the noise (in Hz, i.e., samples/second).
    noise_shape : str
        The shape of the noise. Can be one of 'laplace' (default) or 'gaussian'.

    Returns
    -------
    array
        Vector containing the distorted signal.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=10, num=10000))
    >>> signals = pd.DataFrame({
            "Freq100": nk.signal_distord(signal, noise_frequency=200),
            "Freq50": nk.signal_distord(signal, noise_frequency=50),
            "Freq10": nk.signal_distord(signal, noise_frequency=10),
            "Freq5": nk.signal_distord(signal, noise_frequency=5),
            "Raw": signal})
    >>> signals.plot()
    >>>
    >>> distorted = nk.signal_distord(signal, noise_amplitude=[0.3, 0.2, 0.1], noise_frequency=[5, 10, 20], powerline_amplitude=0.05)
    >>> nk.signal_plot(distorted)
    """
    # Basic noise
    noise = _signal_distord_noise_multifrequency(signal,
                                                 signal_sd=np.std(signal, ddof=1),
                                                 sampling_rate=sampling_rate,
                                                 noise_amplitude=noise_amplitude,
                                                 noise_frequency=noise_frequency,
                                                 noise_shape=noise_shape)

    # Powerline noise
    if powerline_amplitude > 0:
        noise += _signal_distord_powerline(signal,
                                           signal_sd=np.std(signal, ddof=1),
                                           sampling_rate=sampling_rate,
                                           powerline_amplitude=powerline_amplitude,
                                           powerline_frequency=powerline_frequency)
    distorted = signal + noise

    return distorted







# =============================================================================
# Internals
# =============================================================================

def _signal_distord_powerline(signal, signal_sd=None, sampling_rate=1000, powerline_frequency=50, powerline_amplitude=0.1):
    freqs = list(np.arange(powerline_frequency, sampling_rate, powerline_frequency))
    noise = _signal_distord_noise_multifrequency(signal,
                                             signal_sd=signal_sd,
                                             sampling_rate=sampling_rate,
                                             noise_amplitude=powerline_amplitude,
                                             noise_frequency=freqs,
                                             noise_shape="gaussian")
    return noise




def _signal_distord_noise_multifrequency(signal, signal_sd=None, sampling_rate=1000, noise_amplitude=0.1, noise_frequency=100, noise_shape="laplace"):
    duration = len(signal) / sampling_rate

    noise = np.zeros(len(signal))
    params = listify(noise_amplitude=noise_amplitude, noise_frequency=noise_frequency, noise_shape=noise_shape)
    for i in range(len(params["noise_amplitude"])):
        if params["noise_frequency"][i] <= sampling_rate:  # Skip noise of higher freq than recording

            # Parameters
            noise_duration = int(duration * params["noise_frequency"][i])
            if signal_sd is None:
                amplitude = params["noise_amplitude"][i]
            else:
                amplitude = params["noise_amplitude"][i] * signal_sd
            shape = params["noise_shape"][i]

            # Generate noise
            noise = _signal_distord_noise(signal, noise_duration, amplitude, shape)
            noise += noise
    return noise



def _signal_distord_noise(signal, noise_duration, noise_amplitude=0.1, noise_shape="laplace"):

    if noise_shape in ["normal", "gaussian"]:
        noise = np.random.normal(0, noise_amplitude, noise_duration)
    elif noise_shape == "laplace":
        noise = np.random.laplace(0, noise_amplitude, noise_duration)
    else:
        raise ValueError("NeuroKit error: signal_distord(): 'noise_shape' "
                         "should be one of 'gaussian' or 'laplace'.")

    noise = signal_resample(noise, desired_length=len(signal), method="interpolation")
    return noise

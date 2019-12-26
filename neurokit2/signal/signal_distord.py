# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .signal_resample import signal_resample
from ..misc import listify


def signal_distord(signal, sampling_rate=1000, noise_amplitude=0.1, noise_frequency=100, noise_shape="laplace"):
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
    >>> signal = np.cos(np.linspace(start=0, stop=10, num=1000))
    >>> signals = pd.DataFrame({
            "Freq100": nk.signal_distord(signal, noise_frequency=100),
            "Freq50": nk.signal_distord(signal, noise_frequency=50),
            "Freq10": nk.signal_distord(signal, noise_frequency=10),
            "Raw": signal})
    >>> signals.plot()
    >>>
    >>> distorted = nk.signal_distord(signal, noise_amplitude=[0.3, 0.2, 0.1], noise_frequency=[5, 10, 50])
    >>> nk.signal_plot(distorted)
    """
    duration = len(signal) / sampling_rate
    signal_sd = np.std(signal, ddof=1)
    distorted = np.array(signal).copy()

    params = listify(noise_amplitude=noise_amplitude, noise_frequency=noise_frequency, noise_shape=noise_shape)
    for i in range(len(params["noise_amplitude"])):
        # Parameters
        duration = int(duration * params["noise_frequency"][i])
        amplitude = params["noise_amplitude"][i] * signal_sd
        shape = params["noise_shape"][i]

        # Generate noise
        noise = _signal_distord(signal, duration, amplitude, shape)
        distorted += noise

    return distorted










def _signal_distord(signal, noise_duration, noise_amplitude=0.1, noise_shape="laplace"):

    if noise_shape in ["normal", "gaussian"]:
        noise = np.random.normal(0, noise_amplitude, noise_duration)
    elif noise_shape == "laplace":
        noise = np.random.laplace(0, noise_amplitude, noise_duration)
    else:
        raise ValueError("NeuroKit error: signal_distord(): 'noise_shape' "
                         "should be one of 'gaussian' or 'laplace'.")

    noise = signal_resample(noise, desired_length=len(signal), method="interpolation")
    return noise

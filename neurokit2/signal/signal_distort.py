# -*- coding: utf-8 -*-
import numpy as np

from .signal_simulate import signal_simulate
from .signal_resample import signal_resample
from ..misc import listify


def signal_distort(signal, sampling_rate=1000, noise_shape="laplace",
                   noise_amplitude=0, noise_frequency=100,
                   powerline_amplitude=0, powerline_frequency=50,
                   artifacts_amplitude=0, artifacts_frequency=200,
                   random_state=None):
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
    >>> signal = nk.signal_simulate(duration=10, frequency=0.5)
    >>>
    >>> # Noise
    >>> pd.DataFrame({
            "Freq100": nk.signal_distord(signal, noise_frequency=200),
            "Freq50": nk.signal_distord(signal, noise_frequency=50),
            "Freq10": nk.signal_distord(signal, noise_frequency=10),
            "Freq5": nk.signal_distord(signal, noise_frequency=5),
            "Raw": signal}).plot()
    >>>
    >>> # Artifacts
    >>> pd.DataFrame({
            "1Hz": nk.signal_distord(signal, noise_amplitude=0, artifacts_frequency=1, artifacts_amplitude=0.5),
            "5Hz": nk.signal_distord(signal, noise_amplitude=0, artifacts_frequency=5, artifacts_amplitude=0.2),
            "Raw": signal}).plot()
    """
    # Seed the random generator for reproducible results.
    np.random.seed(random_state)

    # Make sure that noise_amplitude is a list.
    if isinstance(noise_amplitude, (int, float)):
        noise_amplitude = [noise_amplitude]

    signal_sd = np.std(signal, ddof=1)
    if signal_sd == 0:
        signal_sd = None

    noise = 0

    # Basic noise.
    if min(noise_amplitude) > 0:
        noise += _signal_distord_noise_multifrequency(signal,
                                                      signal_sd=signal_sd,
                                                      sampling_rate=sampling_rate,
                                                      noise_amplitude=noise_amplitude,
                                                      noise_frequency=noise_frequency,
                                                      noise_shape=noise_shape)

    # Powerline noise.
    if powerline_amplitude > 0:
        noise += _signal_distord_powerline(signal, signal_sd=signal_sd,
                                           sampling_rate=sampling_rate,
                                           powerline_frequency=powerline_frequency,
                                           powerline_amplitude=powerline_amplitude)

    # Artifacts.
    if artifacts_amplitude > 0:
        noise += _signal_distord_artifacts(signal,
                                           signal_sd=signal_sd,
                                           sampling_rate=sampling_rate,
                                           artifacts_frequency=artifacts_frequency,
                                           artifacts_amplitude=artifacts_amplitude)

    distorted = signal + noise

    return distorted


def _signal_distord_artifacts(signal, signal_sd=None, sampling_rate=1000,
                              artifacts_frequency=0, artifacts_amplitude=.1,
                              artifacts_shape="laplace"):

    duration = len(signal) / sampling_rate

    noise_duration = int(duration * artifacts_frequency)
    # Generate oscillatory signal of given frequency.
    # artifacts = signal_simulate(duration=duration, sampling_rate=sampling_rate,
    #                             frequency=artifacts_frequency, amplitude=1)
    artifacts = _signal_distord_noise(signal, noise_duration,
                                      artifacts_amplitude, artifacts_shape)

    # Generate artifact burst with random onset and random duration.
    n_artifacts = 5

    min_duration = int(np.rint(len(artifacts) * .001))
    max_duration = int(np.rint(len(artifacts) * .01))
    artifact_durations = np.random.randint(min_duration, max_duration,
                                           n_artifacts)

    artifact_onsets = np.random.randint(0, len(artifacts) - max_duration,
                                        n_artifacts)
    artifact_offsets = artifact_onsets + artifact_durations

    artifact_idcs = np.array([False] * len(artifacts))
    for i in range(n_artifacts):
        artifact_idcs[artifact_onsets[i]:artifact_offsets[i]] = True

    artifacts[~artifact_idcs] = 0

    # Scale amplitude by the signal's standard deviation.
    if signal_sd is not None:
        artifacts_amplitude *= signal_sd
    artifacts *= artifacts_amplitude

    return artifacts


def _signal_distord_powerline(signal, signal_sd=None, sampling_rate=1000,
                              powerline_frequency=50,
                              powerline_amplitude=.1):

    duration = len(signal) / sampling_rate
    powerline_noise = signal_simulate(duration=duration,
                                      sampling_rate=sampling_rate,
                                      frequency=powerline_frequency,
                                      amplitude=1)

    if signal_sd is not None:
        powerline_amplitude *= signal_sd
    powerline_noise *= powerline_amplitude

    return powerline_noise


def _signal_distord_noise_multifrequency(signal, signal_sd=None,
                                         sampling_rate=1000,
                                         noise_amplitude=.1,
                                         noise_frequency=100,
                                         noise_shape="laplace"):
    duration = len(signal) / sampling_rate
    base_noise = np.zeros(len(signal))
    params = listify(noise_amplitude=noise_amplitude,
                     noise_frequency=noise_frequency, noise_shape=noise_shape)

    for i in range(len(params["noise_amplitude"])):

        freq = params["noise_frequency"][i]
        amp = params["noise_amplitude"][i]
        shape = params["noise_shape"][i]
        # Apply a very conservative Nyquist criterion in order to ensure
        # sufficiently sampled signals.
        nyquist = sampling_rate * .1
        if freq > nyquist:
            raise ValueError(f"NeuroKit error: Please choose frequencies smaller than {nyquist}.")
        # Also make sure that at leat one period of the frequency can be
        # captured over the duration of the signal.
        if (1 / freq) > duration:
            raise ValueError(f"NeuroKit error: Please choose frequencies larger than {1 / duration}.")

        noise_duration = int(duration * freq)

        if signal_sd is not None:
            amp *= signal_sd

        # Make some noise!
        _base_noise = _signal_distord_noise(signal, noise_duration, amp, shape)
        base_noise += _base_noise
    return base_noise


def _signal_distord_noise(signal, noise_duration, noise_amplitude=.1,
                          noise_shape="laplace"):

    if noise_shape in ["normal", "gaussian"]:
        _noise = np.random.normal(0, noise_amplitude, noise_duration)
    elif noise_shape == "laplace":
        _noise = np.random.laplace(0, noise_amplitude, noise_duration)
    else:
        raise ValueError("NeuroKit error: signal_distord(): 'noise_shape' "
                         "should be one of 'gaussian' or 'laplace'.")

    if len(_noise) != len(signal):
        _noise = signal_resample(_noise, desired_length=len(signal),
                                 method="interpolation")
    return _noise

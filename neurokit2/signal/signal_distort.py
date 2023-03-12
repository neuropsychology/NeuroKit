# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..misc import NeuroKitWarning, check_random_state, listify
from .signal_resample import signal_resample
from .signal_simulate import signal_simulate


def signal_distort(
    signal,
    sampling_rate=1000,
    noise_shape="laplace",
    noise_amplitude=0,
    noise_frequency=100,
    powerline_amplitude=0,
    powerline_frequency=50,
    artifacts_amplitude=0,
    artifacts_frequency=100,
    artifacts_number=5,
    linear_drift=False,
    random_state=None,
    silent=False,
):
    """**Signal distortion**

    Add noise of a given frequency, amplitude and shape to a signal.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    noise_shape : str
        The shape of the noise. Can be one of ``"laplace"`` (default) or
        ``"gaussian"``.
    noise_amplitude : float
        The amplitude of the noise (the scale of the random function, relative
        to the standard deviation of the signal).
    noise_frequency : float
        The frequency of the noise (in Hz, i.e., samples/second).
    powerline_amplitude : float
        The amplitude of the powerline noise (relative to the standard
        deviation of the signal).
    powerline_frequency : float
        The frequency of the powerline noise (in Hz, i.e., samples/second).
    artifacts_amplitude : float
        The amplitude of the artifacts (relative to the standard deviation of
        the signal).
    artifacts_frequency : int
        The frequency of the artifacts (in Hz, i.e., samples/second).
    artifacts_number : int
        The number of artifact bursts. The bursts have a random duration
        between 1 and 10% of the signal duration.
    linear_drift : bool
        Whether or not to add linear drift to the signal.
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator. See for ``misc.check_random_state`` for further information.
    silent : bool
        Whether or not to display warning messages.

    Returns
    -------
    array
        Vector containing the distorted signal.

    Examples
    --------
    .. ipython:: python

      import numpy as np
      import pandas as pd
      import neurokit2 as nk

      signal = nk.signal_simulate(duration=10, frequency=0.5)

      # Noise
      @savefig p_signal_distort1.png scale=100%
      noise = pd.DataFrame({"Freq100": nk.signal_distort(signal, noise_frequency=200),
                           "Freq50": nk.signal_distort(signal, noise_frequency=50),
                           "Freq10": nk.signal_distort(signal, noise_frequency=10),
                           "Freq5": nk.signal_distort(signal, noise_frequency=5),
                           "Raw": signal}).plot()
      @suppress
      plt.close()

    .. ipython:: python

      # Artifacts
      @savefig p_signal_distort2.png scale=100%
      artifacts = pd.DataFrame({"1Hz": nk.signal_distort(signal, noise_amplitude=0,
                                                        artifacts_frequency=1,
                                                        artifacts_amplitude=0.5),
                               "5Hz": nk.signal_distort(signal, noise_amplitude=0,
                                                        artifacts_frequency=5,
                                                        artifacts_amplitude=0.2),
                               "Raw": signal}).plot()
      @suppress
      plt.close()

    """
    # Seed the random generator for reproducible results.
    rng = check_random_state(random_state)

    # Make sure that noise_amplitude is a list.
    if isinstance(noise_amplitude, (int, float)):
        noise_amplitude = [noise_amplitude]

    signal_sd = np.std(signal, ddof=1)
    if signal_sd == 0:
        signal_sd = None

    noise = 0

    # Basic noise.
    if min(noise_amplitude) > 0:
        noise += _signal_distort_noise_multifrequency(
            signal,
            signal_sd=signal_sd,
            sampling_rate=sampling_rate,
            noise_amplitude=noise_amplitude,
            noise_frequency=noise_frequency,
            noise_shape=noise_shape,
            silent=silent,
            rng=rng,
        )

    # Powerline noise.
    if powerline_amplitude > 0:
        noise += _signal_distort_powerline(
            signal,
            signal_sd=signal_sd,
            sampling_rate=sampling_rate,
            powerline_frequency=powerline_frequency,
            powerline_amplitude=powerline_amplitude,
            silent=silent,
        )

    # Artifacts.
    if artifacts_amplitude > 0:
        noise += _signal_distort_artifacts(
            signal,
            signal_sd=signal_sd,
            sampling_rate=sampling_rate,
            artifacts_frequency=artifacts_frequency,
            artifacts_amplitude=artifacts_amplitude,
            artifacts_number=artifacts_number,
            silent=silent,
            rng=rng,
        )

    if linear_drift:
        noise += _signal_linear_drift(signal)

    distorted = signal + noise

    return distorted


# ===========================================================================
# Types of Noise
# ===========================================================================


def _signal_linear_drift(signal):

    n_samples = len(signal)
    linear_drift = np.arange(n_samples) * (1 / n_samples)

    return linear_drift


def _signal_distort_artifacts(
    signal,
    signal_sd=None,
    sampling_rate=1000,
    artifacts_frequency=0,
    artifacts_amplitude=0.1,
    artifacts_number=5,
    artifacts_shape="laplace",
    silent=False,
    rng=None,
):

    # Generate artifact burst with random onset and random duration.
    artifacts = _signal_distort_noise(
        len(signal),
        sampling_rate=sampling_rate,
        noise_frequency=artifacts_frequency,
        noise_amplitude=artifacts_amplitude,
        noise_shape=artifacts_shape,
        silent=silent,
        rng=rng,
    )
    if artifacts.sum() == 0:
        return artifacts

    min_duration = int(np.rint(len(artifacts) * 0.001))
    max_duration = int(np.rint(len(artifacts) * 0.01))
    artifact_durations = rng.choice(range(min_duration, max_duration), size=artifacts_number)

    artifact_onsets = rng.choice(len(artifacts) - max_duration, size=artifacts_number)
    artifact_offsets = artifact_onsets + artifact_durations

    artifact_idcs = np.array([False] * len(artifacts))
    for i in range(artifacts_number):
        artifact_idcs[artifact_onsets[i] : artifact_offsets[i]] = True

    artifacts[~artifact_idcs] = 0

    # Scale amplitude by the signal's standard deviation.
    if signal_sd is not None:
        artifacts_amplitude *= signal_sd
    artifacts *= artifacts_amplitude

    return artifacts


def _signal_distort_powerline(
    signal,
    signal_sd=None,
    sampling_rate=1000,
    powerline_frequency=50,
    powerline_amplitude=0.1,
    silent=False,
):

    duration = len(signal) / sampling_rate
    powerline_noise = signal_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        frequency=powerline_frequency,
        amplitude=1,
        silent=silent,
    )

    if signal_sd is not None:
        powerline_amplitude *= signal_sd
    powerline_noise *= powerline_amplitude

    return powerline_noise


def _signal_distort_noise_multifrequency(
    signal,
    signal_sd=None,
    sampling_rate=1000,
    noise_amplitude=0.1,
    noise_frequency=100,
    noise_shape="laplace",
    silent=False,
    rng=None,
):
    base_noise = np.zeros(len(signal))
    params = listify(
        noise_amplitude=noise_amplitude, noise_frequency=noise_frequency, noise_shape=noise_shape
    )

    for i in range(len(params["noise_amplitude"])):

        freq = params["noise_frequency"][i]
        amp = params["noise_amplitude"][i]
        shape = params["noise_shape"][i]

        if signal_sd is not None:
            amp *= signal_sd

        # Make some noise!
        _base_noise = _signal_distort_noise(
            len(signal),
            sampling_rate=sampling_rate,
            noise_frequency=freq,
            noise_amplitude=amp,
            noise_shape=shape,
            silent=silent,
            rng=rng,
        )
        base_noise += _base_noise

    return base_noise


def _signal_distort_noise(
    n_samples,
    sampling_rate=1000,
    noise_frequency=100,
    noise_amplitude=0.1,
    noise_shape="laplace",
    silent=False,
    rng=None,
):

    _noise = np.zeros(n_samples)
    # Apply a very conservative Nyquist criterion in order to ensure
    # sufficiently sampled signals.
    nyquist = sampling_rate * 0.1
    if noise_frequency > nyquist:
        if not silent:
            warn(
                f"Skipping requested noise frequency "
                f" of {noise_frequency} Hz since it cannot be resolved at "
                f" the sampling rate of {sampling_rate} Hz. Please increase "
                f" sampling rate to {noise_frequency * 10} Hz or choose "
                f" frequencies smaller than or equal to {nyquist} Hz.",
                category=NeuroKitWarning,
            )
        return _noise
    # Also make sure that at least one period of the frequency can be
    # captured over the duration of the signal.
    duration = n_samples / sampling_rate
    if (1 / noise_frequency) > duration:
        if not silent:
            warn(
                f"Skipping requested noise frequency "
                f" of {noise_frequency} Hz since its period of {1 / noise_frequency} "
                f" seconds exceeds the signal duration of {duration} seconds. "
                f" Please choose noise frequencies larger than "
                f" {1 / duration} Hz or increase the duration of the "
                f" signal above {1 / noise_frequency} seconds.",
                category=NeuroKitWarning,
            )
        return _noise

    noise_duration = int(duration * noise_frequency)

    if noise_shape in ["normal", "gaussian"]:
        _noise = rng.normal(0, noise_amplitude, noise_duration)
    elif noise_shape == "laplace":
        _noise = rng.laplace(0, noise_amplitude, noise_duration)
    else:
        raise ValueError(
            "NeuroKit error: signal_distort(): 'noise_shape' should be one of 'gaussian' or 'laplace'."
        )

    if len(_noise) != n_samples:
        _noise = signal_resample(_noise, desired_length=n_samples, method="interpolation")
    return _noise

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
from neurokit2.signal import signal_distort

# Naive phenomenological approximation of PPG.
# The PPG wave is described with four landmarks: wave onset, location of the
# systolic peak, location of the dicrotic notch and location of the diastolic
# peaks. These landmarks are defined based on their x and y coordinates (in a
# time series). These coordinates are then interpolated at the desired
# sampling rate to obtain the PPG signal.


def ppg_simulate(heartrate=70, duration=120, sfreq=1000,
                 frequency_modulation=.4, show=False, random_state=None):

    # At the requested sampling rate, how long is a period at the requested
    # heart-rate and how often does that period fit into the requested
    # duration?
    period = 60 / heartrate   # in seconds
    n_period = int(np.floor(duration / period))
    periods = np.ones(n_period) * period
    # Seconds at which waves begin.
    x_onset = np.cumsum(periods)
    # Add respiratory sinus arrythmia (frequency modulation).
    periods, x_onset = _frequency_modulation(x_onset, periods,
                                             modulation_frequency=.05,
                                             modulation_strength=frequency_modulation)
    # Corresponding signal amplitudes.
    y_onset = 0

    # Seconds at which the systolic peaks occur within the waves.
    x_sys = x_onset + 0.175 * periods
    # Corresponding signal amplitudes.
    y_sys = y_onset + 1.5    # express as percentage of y_onset

    # Seconds at which the dicrotic notches occur within the waves.
    x_notch = x_onset + 0.4 * periods
    # Corresponding signal amplitudes.
    y_notch = y_onset + y_sys * .5    # express as percentage of y_onset

    # Seconds at which the diatolic peaks occur within the waves.
    x_dia = x_onset + 0.45 * periods
    # Corresponding signal amplitudes.
    y_dia = y_onset + y_sys * .51    # express as percentage of y_onset

    x_all = np.concatenate((x_onset, x_sys, x_notch, x_dia))
    x_all.sort(kind="mergesort")
    x_all = np.rint(x_all * sfreq).astype(int)    # convert seconds to samples
    y_all = [y_onset, y_sys, y_notch, y_dia] * n_period

    # Interpolate a continuous signal between the landmarks (i.e., Cartesian
    # coordinates).
    f = Akima1DInterpolator(x_all, y_all)
    samples = np.arange(0, duration * sfreq)
    ppg = f(samples)
    # Remove NANs (values outside interpolation range, i.e., after last sample).
    ppg = ppg[~np.isnan(ppg)]

    # Add motion artifacts, baseline drift, and powerline noise.
    drift = .05
    motion = .5
    ppg_noisy = signal_distort(ppg, sampling_rate=sfreq,
                               noise_amplitude=[1, .5],
                               noise_frequency=[drift, motion],
                               powerline_amplitude=.1, powerline_frequency=50,
                               artifacts_amplitude=1, artifacts_frequency=100,
                               random_state=random_state)

    if show:
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax0.scatter(x_all, y_all, c="r")
        ax0.plot(ppg)
        ax1.plot(ppg_noisy)

    return ppg_noisy


def _frequency_modulation(seconds, periods, modulation_frequency,
                          modulation_strength):
    """
    modulator_frequency = .1    # in Hz, RSA modulation
    modulator_strength = .5    # must be between 0 and 1
    """

    # Apply a very conservative Nyquist criterion.
    nyquist = (1 / periods[0]) * .1
    if modulation_frequency > nyquist:
        print(f"Please choose a modulation frequency lower than {nyquist}.")
        return

    modulator = modulation_strength * np.sin(2 * np.pi * modulation_frequency *
                                             seconds) + 1
    periods_modulated = periods * modulator
    seconds_modulated = np.cumsum(periods_modulated)
    seconds_modulated -= seconds_modulated[0]    # make sure seconds start at zero

    return periods_modulated, seconds_modulated


def _amplitude_modulation():
    pass

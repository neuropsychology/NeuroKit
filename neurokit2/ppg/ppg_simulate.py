# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator, Akima1DInterpolator
from neurokit2.signal import signal_distort

# Naive approximation of PPG
# The PPG wave is described with four landmarks: wave onset, location of the
# systolic peak, location of the dicrotic notch and location of the diastolic
# peaks. These landmarks are defined based on their x and y coordinates (in a
# time series). These coordinates are then interpolated at the desired
# sampling rate to obtain the PPG signal.


def ppg_simulate(heartrate=70, duration=120, sfreq=100):

    # at the requested sampling rate, how long is a period at the requested heart
    # rate and how often does that period fit into the requested duration?
    period = 60 / heartrate   # in seconds
    n_period = int(np.floor(duration / period))
    periods = np.ones(n_period) * period
    # seconds at which periods end
    x_onset = np.cumsum(periods)
    periods, x_onset = _frequency_modulation(x_onset, periods,
                                             modulation_frequency=.05,
                                             modulation_strength=.2)
    y_onset = 0    # in volt Mesin et al., 2019

    x_sys = x_onset + 0.175 * periods
    y_sys = y_onset + 1.5    # express as percentage of y_onset

    x_notch = x_onset + 0.4 * periods
    y_notch = y_onset + y_sys * .5    # express as percentage of y_onset

    x_dia = x_onset + 0.45 * periods
    y_dia = y_onset + y_sys * .51    # express as percentage of y_onset

    x_all = np.concatenate((x_onset, x_sys, x_notch, x_dia))
    x_all.sort(kind="mergesort")
    x_all = np.rint(x_all * sfreq).astype(int)
    y_all = [y_onset, y_sys, y_notch, y_dia] * n_period

    # f_cubic = interp1d(x_all, y_all, kind="cubic", bounds_error=False,
    #                    fill_value=([y_all[0]], [y_all[-1]]))
    # f_hermite = PchipInterpolator(x_all, y_all, extrapolate=False)
    f_akima = Akima1DInterpolator(x_all, y_all)

    samples = np.arange(0, duration * sfreq)

    # ppg_cubic = f_cubic(samples)
    # ppg_hermite = f_hermite(samples)
    ppg_akima = f_akima(samples)

    ppg_akima = signal_distort(ppg_akima, sampling_rate=sfreq,
                               powerline_amplitude=.1)


    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    # ax0.scatter(x_onset, y_onset)
    # ax0.scatter(x_sys, y_sys)
    # ax0.scatter(x_notch, y_notch)
    # ax0.scatter(x_dia, y_dia)
    ax0.scatter(x_all, y_all, c="r")
    # ax0.plot(ppg_cubic, c="g")
    ax0.plot(ppg_akima, c="m")
    # ax0.plot(ppg_hermite, c="k")
    # ax0.plot(x_periods, modulator)
    # ax1.scatter(x_periods_modulated, amp)

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


def _add_noise():
    pass


def _add_baselinedrift():
    pass

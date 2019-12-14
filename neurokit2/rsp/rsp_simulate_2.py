# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def resp_simulate_data(nCycles=100,
                       sampling_rate=1000,
                       breathing_rate=0.25,
                       average_amplitude=0.5,
                       amplitude_variance=0.1,
                       phase_variance=0.1,
                       inhale_pause_percent=0.3,
                       inhale_pause_avgLength=0.2,
                       inhale_pauseLength_variance=0.5,
                       exhale_pause_percent=0.3,
                       exhale_pause_avgLength=0.2,
                       exhale_pauseLength_variance=0.5,
                       pause_amplitude=0.1,
                       pause_amplitude_variance=0.2,
                       signal_noise=0.1):
    """
    Simulates a recording of human airflow data by appending individually constructed sin waves and pauses in sequence.

    Parameters
    ----------
    nCycles : int or float
        number of breathing cycles to simulate.
    sampling_rate :
        sampling rate.
    breathing_rate :
        average breathing rate.
    average_amplitude :
        average amplitude of inhales and exhales.
    amplitude_variance:
        variance in respiratory amplitudes.
    phase_variance:
        variance in duration of individual breaths.
    inhale_pause_percent :
        percent of inhales followed by a pause.
    inhale_pause_avgLength :
        average length of inhale pauses.
    inhale_pauseLength_variance :
        variance in inhale pause length.
    exhale_pause_percent :
        percent of exhales followed by a pause.
    exhale_pause_avgLength :
        average length of exhale pauses.
    exhale_pauseLength_variance :
        variance in exhale pause length.
    pause_amplitude :
        noise amplitude of pauses.
    pause_amplitude_variance :
        variance in pause noise.
    signal_noise :
        percent of noise saturation in the simulated signal.

    Returns
    ----------

    """
    # Define additional parameters
    sample_phase = sampling_rate / breathing_rate
    inhale_pause_phase = np.round(inhale_pause_avgLength *
                                  sample_phase).astype(int)
    exhale_pause_phase = np.round(exhale_pause_avgLength *
                                  sample_phase).astype(int)

    # Normalize variance by average breath amplitude
    amplitude_variance_normed = average_amplitude * amplitude_variance
    amplitudes__with_noise = np.random.randn(
        nCycles) * amplitude_variance_normed + average_amplitude

    # Normalize phase by average breath length
    phase_variance_normed = phase_variance * sample_phase
    phases_with_noise = np.round(
        np.random.randn(nCycles) * phase_variance_normed +
        sample_phase).astype(int)

    # Normalize pause lengths by phase and variation
    inhale_pauseLength_variance_normed = inhale_pause_phase * inhale_pauseLength_variance
    inhale_pauseLengths_with_noise = np.round(
        np.random.randn(nCycles) * inhale_pauseLength_variance_normed +
        inhale_pause_phase).astype(int)
    exhale_pauseLength_variance_normed = exhale_pause_phase * exhale_pauseLength_variance
    exhale_pauseLengths_with_noise = np.round(
        np.random.randn(nCycles) * exhale_pauseLength_variance_normed +
        inhale_pause_phase).astype(int)  ##why inhale pause phase??????????????

    # Normalize pause amplitudes
    pause_amplitude_variance_normed = pause_amplitude * pause_amplitude_variance

    # Initialize empty vector to fill with simulated data
    simulated_respiration = []

    # Initialize parameters to save
    inhale_onsets = np.zeros(nCycles)
    exhale_onsets = np.zeros(nCycles)

    inhale_pause_onsets = np.zeros(nCycles)
    exhale_pause_onsets = np.zeros(nCycles)

    inhale_lengths = np.zeros(nCycles)
    inhale_pauseLengths = np.zeros(nCycles)
    exhale_lengths = np.zeros(nCycles)
    exhale_pauseLengths = np.zeros(nCycles)

    inhale_peaks = np.zeros(nCycles)
    exhale_troughs = np.zeros(nCycles)

    i = 1
    for c in range(nCycles):
        # Determine length of inhale pause for this cycle
        if np.random.rand() < inhale_pause_percent:
            this_inhale_pauseLength = inhale_pauseLengths_with_noise[c]
            this_inhale_pause = np.random.randn(
                this_inhale_pauseLength) * pause_amplitude_variance_normed
        else:
            this_inhale_pauseLength = 0
            this_inhale_pause = []

        # Determine length of exhale pause for this cycle
        if np.random.rand() < exhale_pause_percent:
            this_exhale_pauseLength = exhale_pauseLengths_with_noise[c]
            this_exhale_pause = np.random.randn(
                this_exhale_pauseLength) * pause_amplitude_variance_normed
        else:
            this_exhale_pauseLength = 0
            this_exhale_pause = []

        # Determine length of inhale and exhale for this cycle to main breathing rate
        cycle_length = phases_with_noise[c] - (this_inhale_pauseLength +
                                               this_exhale_pauseLength)

        # If pauses are longer than the time alloted for this breath, set them to 0 so a real breath can be simulated. This will deviate the statistics from those initialized but is unavaoidable at the current state
        if (cycle_length <= 0) or (cycle_length < min(phases_with_noise) / 4):
            this_inhale_pauseLength = 0
            this_inhale_pause = []
            this_exhale_pauseLength = 0
            this_exhale_pause = []
            cycle_length = phases_with_noise[c] - (this_inhale_pauseLength +
                                                   this_exhale_pauseLength)

        # Compute inhale and exhale for this cycle
        this_cycle = np.sin(np.linspace(
            0, 2 * np.pi, cycle_length)) * amplitudes__with_noise[c]
        half_cycle = np.round(len(this_cycle) / 2).astype(int)
        this_inhale = this_cycle[0:half_cycle]
        this_inhale_length = len(this_inhale)
        this_exhale = this_cycle[half_cycle:]
        this_exhale_length = len(this_exhale)

        # Save parameters for checking
        inhale_lengths[c] = this_inhale_length
        inhale_pauseLengths[c] = this_inhale_pauseLength
        exhale_lengths[c] = this_exhale_length
        exhale_pauseLengths[c] = this_exhale_pauseLength
        inhale_onsets[c] = i
        exhale_onsets[c] = i + this_inhale_length + this_inhale_pauseLength

        if len(this_inhale_pause) > 0:
            inhale_pause_onsets[c] = i + this_inhale_length
        else:
            inhale_pause_onsets[c] = np.nan

        if len(this_exhale_pause) > 0:
            exhale_pause_onsets[
                c] = i + this_inhale_length + this_inhale_pauseLength + this_exhale_length
        else:
            exhale_pause_onsets[c] = np.nan

        # Compose breath from parameters
        this_breath = np.hstack(
            [this_inhale, this_inhale_pause, this_exhale, this_exhale_pause])

        # Compute max flow for inhale and exhale for this breath
        max_ID = np.argmax(this_breath)
        min_ID = np.argmin(this_breath)
        inhale_peaks[c] = i + max_ID
        exhale_troughs[c] = i + min_ID

        # Append breath to simulated resperation vector
        simulated_respiration = np.hstack([simulated_respiration, this_breath])
        i = i + len(this_breath) - 1

    if signal_noise == 0:
        signal_noise = 0.0001

    noise_vector = np.random.rand(
        *simulated_respiration.shape) * average_amplitude
    simulated_respiration = simulated_respiration * (
        1 - signal_noise) + noise_vector * signal_noise
    raw_features = {
        'Inhale Onsets': inhale_onsets,
        'Exhale Onsets': exhale_onsets,
        'Inhale Pause Onsets': inhale_pause_onsets,
        'Exhale Pause Onsets': exhale_pause_onsets,
        'Inhale Lengths': inhale_lengths / sampling_rate,
        'Inhale Pause Lengths': inhale_pauseLengths / sampling_rate,
        'Exhale Lengths': exhale_lengths / sampling_rate,
        'Exhale Pause Lengths': exhale_pauseLengths / sampling_rate,
        'Inhale Peaks': inhale_peaks,
        'Exhale Troughs': exhale_troughs
    }
    if len(inhale_pauseLengths[inhale_pauseLengths > 0]) > 0:
        avg_inhale_pauseLength = np.mean(
            inhale_pauseLengths[inhale_pauseLengths > 0])
    else:
        avg_inhale_pauseLength = 0

    if len(exhale_pauseLengths[exhale_pauseLengths > 0]) > 0:
        avg_exhale_pauseLength = np.mean(
            exhale_pauseLengths[exhale_pauseLengths > 0])
    else:
        avg_exhale_pauseLength = 0

    estimated_breathing_rate = (
        1 / np.mean(np.diff(inhale_onsets))) * sampling_rate
    feature_stats = {
        'Breathing Rate': estimated_breathing_rate,
        'Average Inhale Length': np.mean(inhale_lengths / sampling_rate),
        'Average Inhale Pause Length': avg_inhale_pauseLength / sampling_rate,
        'Average Exhale Length': np.mean(exhale_lengths / sampling_rate),
        'Average Exhale Pause Length': avg_exhale_pauseLength / sampling_rate
    }

    return simulated_respiration, raw_features, feature_stats


simulated_resp = resp_simulate_data(5)
plt.plot(simulated_resp[0])
plt.show()
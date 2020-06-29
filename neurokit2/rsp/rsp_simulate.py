# -*- coding: utf-8 -*-
import numpy as np

from ..signal import signal_distort, signal_simulate, signal_smooth


def rsp_simulate(
    duration=10,
    length=None,
    sampling_rate=1000,
    noise=0.01,
    respiratory_rate=15,
    method="breathmetrics",
    random_state=None,
):
    """Simulate a respiratory signal.

    Generate an artificial (synthetic) respiratory signal of a given duration
    and rate.

    Parameters
    ----------
    duration : int
        Desired length of duration (s).
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    length : int
        The desired length of the signal (in samples).
    noise : float
        Noise level (amplitude of the laplace noise).
    respiratory_rate : float
        Desired number of breath cycles in one minute.
    method : str
        The model used to generate the signal. Can be 'sinusoidal' for a simulation based on a
        trigonometric sine wave that roughly approximates a single respiratory cycle. If
        'breathmetrics' (default), will use an advanced model desbribed `Noto, et al. (2018)
        <https://github.com/zelanolab/breathmetrics/blob/master/simulateRespiratoryData.m>`_.
    random_state : int
        Seed for the random number generator.

    Returns
    -------
    array
        Vector containing the respiratory signal.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import neurokit2 as nk
    >>>
    >>> rsp1 = nk.rsp_simulate(duration=30, method="sinusoidal")
    >>> rsp2 = nk.rsp_simulate(duration=30, method="breathmetrics")
    >>> fig = pd.DataFrame({"RSP_Simple": rsp1, "RSP_Complex": rsp2}).plot(subplots=True)
    >>> fig #doctest: +SKIP

    References
    ----------
    Noto, T., Zhou, G., Schuele, S., Templer, J., & Zelano, C. (2018). Automated analysis of breathing
    waveforms using BreathMetrics: A respiratory signal processing toolbox. Chemical Senses, 43(8), 583–597.
    https://doi.org/10.1093/chemse/bjy045

    See Also
    --------
    rsp_clean, rsp_findpeaks, signal_rate, rsp_process, rsp_plot

    """
    # Seed the random generator for reproducible results
    np.random.seed(random_state)

    # Generate number of samples automatically if length is unspecified
    if length is None:
        length = duration * sampling_rate

    if method.lower() in ["sinusoidal", "sinus", "simple"]:
        rsp = _rsp_simulate_sinusoidal(
            duration=duration, sampling_rate=sampling_rate, respiratory_rate=respiratory_rate
        )
    else:
        rsp = _rsp_simulate_breathmetrics(
            duration=duration, sampling_rate=sampling_rate, respiratory_rate=respiratory_rate
        )
        rsp = rsp[0:length]

    # Add random noise
    if noise > 0:
        rsp = signal_distort(
            rsp,
            sampling_rate=sampling_rate,
            noise_amplitude=noise,
            noise_frequency=[5, 10, 100],
            noise_shape="laplace",
            random_state=random_state,
            silent=True,
        )

    # Reset random seed (so it doesn't affect global)
    np.random.seed(None)
    return rsp


# =============================================================================
# Simple Sinusoidal Model
# =============================================================================
def _rsp_simulate_sinusoidal(duration=10, sampling_rate=1000, respiratory_rate=15):
    """Generate an artificial (synthetic) respiratory signal by trigonometric sine wave that roughly approximates a
    single respiratory cycle."""
    # Generate values along the length of the duration
    rsp = signal_simulate(
        duration=duration, sampling_rate=sampling_rate, frequency=respiratory_rate / 60, amplitude=0.5
    )

    return rsp


# =============================================================================
# BreathMetrics Model
# =============================================================================
def _rsp_simulate_breathmetrics_original(
    nCycles=100,
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
    signal_noise=0.1,
):
    """Simulates a recording of human airflow data by appending individually constructed sin waves and pauses in
    sequence. This is translated from the matlab code available `here.

    <https://github.com/zelanolab/breathmetrics/blob/master/simulateRespiratoryData.m>`_ by Noto, et al. (2018).

    Parameters
    ----------
    nCycles : int or float
        number of breathing cycles to simulate.
    sampling_rate : int
        sampling rate.
    breathing_rate : float
        average breathing rate.
    average_amplitude : float
        average amplitude of inhales and exhales.
    amplitude_variance: float
        variance in respiratory amplitudes.
    phase_variance: float
        variance in duration of individual breaths.
    inhale_pause_percent : float
        percent of inhales followed by a pause.
    inhale_pause_avgLength : float
        average length of inhale pauses.
    inhale_pauseLength_variance : float
        variance in inhale pause length.
    exhale_pause_percent : float
        percent of exhales followed by a pause.
    exhale_pause_avgLength : float
        average length of exhale pauses.
    exhale_pauseLength_variance : float
        variance in exhale pause length.
    pause_amplitude : float
        noise amplitude of pauses.
    pause_amplitude_variance : float
        variance in pause noise.
    signal_noise : float
        percent of noise saturation in the simulated signal.

    Returns
    ----------
    signal
        vector containing breathmetrics simulated rsp signal.

    """
    # Define additional parameters
    sample_phase = sampling_rate / breathing_rate
    inhale_pause_phase = np.round(inhale_pause_avgLength * sample_phase).astype(int)
    exhale_pause_phase = np.round(exhale_pause_avgLength * sample_phase).astype(int)

    # Normalize variance by average breath amplitude
    amplitude_variance_normed = average_amplitude * amplitude_variance
    amplitudes_with_noise = np.random.randn(nCycles) * amplitude_variance_normed + average_amplitude
    amplitudes_with_noise[amplitudes_with_noise < 0] = 0

    # Normalize phase by average breath length
    phase_variance_normed = phase_variance * sample_phase
    phases_with_noise = np.round(np.random.randn(nCycles) * phase_variance_normed + sample_phase).astype(int)
    phases_with_noise[phases_with_noise < 0] = 0

    # Normalize pause lengths by phase and variation
    inhale_pauseLength_variance_normed = inhale_pause_phase * inhale_pauseLength_variance
    inhale_pauseLengths_with_noise = np.round(
        np.random.randn(nCycles) * inhale_pauseLength_variance_normed + inhale_pause_phase
    ).astype(int)
    inhale_pauseLengths_with_noise[inhale_pauseLengths_with_noise < 0] = 0
    exhale_pauseLength_variance_normed = exhale_pause_phase * exhale_pauseLength_variance
    exhale_pauseLengths_with_noise = np.round(
        np.random.randn(nCycles) * exhale_pauseLength_variance_normed + inhale_pause_phase
    ).astype(int)

    # why inhale pause phase?
    exhale_pauseLengths_with_noise[exhale_pauseLengths_with_noise < 0] = 0

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
            this_inhale_pause = np.random.randn(this_inhale_pauseLength) * pause_amplitude_variance_normed
            this_inhale_pause[this_inhale_pause < 0] = 0
        else:
            this_inhale_pauseLength = 0
            this_inhale_pause = []

        # Determine length of exhale pause for this cycle
        if np.random.rand() < exhale_pause_percent:
            this_exhale_pauseLength = exhale_pauseLengths_with_noise[c]
            this_exhale_pause = np.random.randn(this_exhale_pauseLength) * pause_amplitude_variance_normed
            this_exhale_pause[this_exhale_pause < 0] = 0
        else:
            this_exhale_pauseLength = 0
            this_exhale_pause = []

        # Determine length of inhale and exhale for this cycle to main
        # breathing rate
        cycle_length = phases_with_noise[c] - (this_inhale_pauseLength + this_exhale_pauseLength)

        # If pauses are longer than the time alloted for this breath, set them
        # to 0 so a real breath can be simulated. This will deviate the
        # statistics from those initialized but is unavaoidable at the current
        # state
        if (cycle_length <= 0) or (cycle_length < min(phases_with_noise) / 4):
            this_inhale_pauseLength = 0
            this_inhale_pause = []
            this_exhale_pauseLength = 0
            this_exhale_pause = []
            cycle_length = phases_with_noise[c] - (this_inhale_pauseLength + this_exhale_pauseLength)

        # Compute inhale and exhale for this cycle
        this_cycle = np.sin(np.linspace(0, 2 * np.pi, cycle_length)) * amplitudes_with_noise[c]
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
            exhale_pause_onsets[c] = i + this_inhale_length + this_inhale_pauseLength + this_exhale_length
        else:
            exhale_pause_onsets[c] = np.nan

        # Compose breath from parameters
        this_breath = np.hstack([this_inhale, this_inhale_pause, this_exhale, this_exhale_pause])

        # Compute max flow for inhale and exhale for this breath
        max_ID = np.argmax(this_breath)
        min_ID = np.argmin(this_breath)
        inhale_peaks[c] = i + max_ID
        exhale_troughs[c] = i + min_ID

        # Append breath to simulated resperation vector
        simulated_respiration = np.hstack([simulated_respiration, this_breath])
        i = i + len(this_breath) - 1

    # Smooth signal
    simulated_respiration = signal_smooth(simulated_respiration, kernel="boxzen", size=sampling_rate / 2)

    if signal_noise == 0:
        signal_noise = 0.0001

    noise_vector = np.random.rand(*simulated_respiration.shape) * average_amplitude
    simulated_respiration = simulated_respiration * (1 - signal_noise) + noise_vector * signal_noise
    raw_features = {
        "Inhale Onsets": inhale_onsets,
        "Exhale Onsets": exhale_onsets,
        "Inhale Pause Onsets": inhale_pause_onsets,
        "Exhale Pause Onsets": exhale_pause_onsets,
        "Inhale Lengths": inhale_lengths / sampling_rate,
        "Inhale Pause Lengths": inhale_pauseLengths / sampling_rate,
        "Exhale Lengths": exhale_lengths / sampling_rate,
        "Exhale Pause Lengths": exhale_pauseLengths / sampling_rate,
        "Inhale Peaks": inhale_peaks,
        "Exhale Troughs": exhale_troughs,
    }
    if len(inhale_pauseLengths[inhale_pauseLengths > 0]) > 0:
        avg_inhale_pauseLength = np.mean(inhale_pauseLengths[inhale_pauseLengths > 0])
    else:
        avg_inhale_pauseLength = 0

    if len(exhale_pauseLengths[exhale_pauseLengths > 0]) > 0:
        avg_exhale_pauseLength = np.mean(exhale_pauseLengths[exhale_pauseLengths > 0])
    else:
        avg_exhale_pauseLength = 0

    estimated_breathing_rate = (1 / np.mean(np.diff(inhale_onsets))) * sampling_rate
    feature_stats = {
        "Breathing Rate": estimated_breathing_rate,
        "Average Inhale Length": np.mean(inhale_lengths / sampling_rate),
        "Average Inhale Pause Length": avg_inhale_pauseLength / sampling_rate,
        "Average Exhale Length": np.mean(exhale_lengths / sampling_rate),
        "Average Exhale Pause Length": avg_exhale_pauseLength / sampling_rate,
    }

    return simulated_respiration, raw_features, feature_stats


def _rsp_simulate_breathmetrics(duration=10, sampling_rate=1000, respiratory_rate=15):

    n_cycles = int(respiratory_rate / 60 * duration)

    # Loop until it doesn't fail
    rsp = False
    while rsp is False:
        # Generate a longer than necessary signal so it won't be shorter
        rsp, _, __ = _rsp_simulate_breathmetrics_original(
            nCycles=int(n_cycles * 1.5),
            sampling_rate=sampling_rate,
            breathing_rate=respiratory_rate / 60,
            signal_noise=0,
        )
    return rsp

# -*- coding: utf-8 -*-

import itertools
import numpy as np
import neurokit2 as nk
import pytest


durations = (20, 200)
sampling_rates = (50, 500)
heart_rates = (50, 120)
freq_modulations = (0.1, 0.4)

params = [durations,
          sampling_rates,
          heart_rates,
          freq_modulations]

params_combis = list(itertools.product(*params))

@pytest.mark.parametrize("duration, sampling_rate, heart_rate, freq_modulation",
                         params_combis)
def test_ppg_simulate(duration, sampling_rate, heart_rate, freq_modulation):

    ppg = nk.ppg_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        heart_rate=heart_rate,
        frequency_modulation=freq_modulation,
        ibi_randomness=0,
        drift=0,
        motion_amplitude=0,
        powerline_amplitude=0,
        burst_amplitude=0,
        burst_number=0,
        random_state=42,
        show=False,
    )

    assert ppg.size == duration * sampling_rate

    signals, _ = nk.ppg_process(ppg, sampling_rate=sampling_rate)
    assert np.allclose(signals["PPG_Rate"].mean(), heart_rate, atol=1)

    # Ensure that the heart rate fluctuates in the requested range.
    groundtruth_range = freq_modulation * heart_rate
    observed_range = np.percentile(signals['PPG_Rate'], 90) - np.percentile(signals['PPG_Rate'], 10)
    assert np.allclose(groundtruth_range, observed_range, atol=groundtruth_range * .15)

    # TODO: test influence of different noise configurations


@pytest.mark.parametrize("ibi_randomness, std_heart_rate",
                         [(.1, 3), (.2, 5), (.3, 8), (.4, 11), (.5, 14), (.6, 19)])
def test_ppg_simulate_ibi(ibi_randomness, std_heart_rate):

    ppg = nk.ppg_simulate(
        duration=20,
        sampling_rate=50,
        heart_rate=70,
        frequency_modulation=0,
        ibi_randomness=ibi_randomness,
        drift=0,
        motion_amplitude=0,
        powerline_amplitude=0,
        burst_amplitude=0,
        burst_number=0,
        random_state=42,
        show=False,
    )

    assert ppg.size == 20 * 50

    signals, _ = nk.ppg_process(ppg, sampling_rate=50)
    assert np.allclose(signals["PPG_Rate"].mean(), 70, atol=1.5)

    # Ensure that standard deviation of heart rate
    assert np.allclose(signals["PPG_Rate"].std(), std_heart_rate, atol=1)


    # TODO: test influence of different noise configurations


def test_ppg_clean():

    sampling_rate = 500

    ppg = nk.ppg_simulate(
        duration=30,
        sampling_rate=sampling_rate,
        heart_rate=180,
        frequency_modulation=0.01,
        ibi_randomness=0.1,
        drift=1,
        motion_amplitude=0.5,
        powerline_amplitude=0.1,
        burst_amplitude=1,
        burst_number=5,
        random_state=42,
        show=False,
    )
    ppg_cleaned_elgendi = nk.ppg_clean(ppg, sampling_rate=sampling_rate, method="elgendi")

    assert ppg.size == ppg_cleaned_elgendi.size

    # Assert that bandpass filter with .5 Hz lowcut and 8 Hz highcut was applied.
    fft_raw = np.abs(np.fft.rfft(ppg))
    fft_elgendi = np.abs(np.fft.rfft(ppg_cleaned_elgendi))

    freqs = np.fft.rfftfreq(ppg.size, 1 / sampling_rate)

    assert np.sum(fft_raw[freqs < 0.5]) > np.sum(fft_elgendi[freqs < 0.5])
    assert np.sum(fft_raw[freqs > 8]) > np.sum(fft_elgendi[freqs > 8])


def test_ppg_findpeaks():

    sampling_rate = 500

    ppg = nk.ppg_simulate(
        duration=30,
        sampling_rate=sampling_rate,
        heart_rate=60,
        frequency_modulation=0.01,
        ibi_randomness=0.1,
        drift=1,
        motion_amplitude=0.5,
        powerline_amplitude=0.1,
        burst_amplitude=1,
        burst_number=5,
        random_state=42,
        show=True,
    )
    ppg_cleaned_elgendi = nk.ppg_clean(ppg, sampling_rate=sampling_rate, method="elgendi")

    info_elgendi = nk.ppg_findpeaks(ppg_cleaned_elgendi, sampling_rate=sampling_rate, show=True)

    peaks = info_elgendi["PPG_Peaks"]

    assert peaks.size == 29
    assert peaks.sum() == 219763

# -*- coding: utf-8 -*-

import numpy as np

import neurokit2 as nk


def test_ppg_simulate():

    ppg1 = nk.ppg_simulate(duration=20, sampling_rate=500, heart_rate=70,
                           frequency_modulation=.3, ibi_randomness=.25,
                           drift=1, motion_amplitude=.5,
                           powerline_amplitude=.1, burst_amplitude=1,
                           burst_number=5, random_state=42, show=False)
    assert ppg1.size == 20 * 500

    ppg2 = nk.ppg_simulate(duration=200, sampling_rate=1000, heart_rate=70,
                           frequency_modulation=.3, ibi_randomness=.25,
                           drift=1, motion_amplitude=.5,
                           powerline_amplitude=.1, burst_amplitude=1,
                           burst_number=5, random_state=42, show=False)
    assert ppg2.size == 200 * 1000

    # Ensure that frequency_modulation does not affect other signal properties.
    ppg3 = nk.ppg_simulate(duration=200, sampling_rate=1000, heart_rate=70,
                           frequency_modulation=1, ibi_randomness=.25,
                           drift=1, motion_amplitude=.5,
                           powerline_amplitude=.1, burst_amplitude=1,
                           burst_number=5, random_state=42, show=False)
    assert np.allclose((ppg2.mean() - ppg3.mean()), 0, atol=1e-2)
    assert np.allclose((ppg2.std() - ppg3.std()), 0, atol=1e-2)

    # Ensure that ibi_randomness does not affect other signal properties.
    ppg4 = nk.ppg_simulate(duration=200, sampling_rate=1000, heart_rate=70,
                           frequency_modulation=1, ibi_randomness=1,
                           drift=1, motion_amplitude=.5,
                           powerline_amplitude=.1, burst_amplitude=1,
                           burst_number=5, random_state=42, show=False)
    assert np.allclose((ppg3.mean() - ppg4.mean()), 0, atol=1e-1)
    assert np.allclose((ppg3.std() - ppg4.std()), 0, atol=1e-1)

    # TODO: test influence of different noise configurations

def test_ppg_clean():

    sampling_rate = 500

    ppg = nk.ppg_simulate(duration=30, sampling_rate=sampling_rate,
                          heart_rate=180, frequency_modulation=.01,
                          ibi_randomness=.1, drift=1, motion_amplitude=.5,
                          powerline_amplitude=.1, burst_amplitude=1,
                          burst_number=5, random_state=42, show=False)
    ppg_cleaned_elgendi = nk.ppg_clean(ppg, sampling_rate=sampling_rate,
                                       method="elgendi")

    assert ppg.size == ppg_cleaned_elgendi.size

    # Assert that bandpass filter with .5 Hz lowcut and 8 Hz highcut was applied.
    fft_raw = np.abs(np.fft.rfft(ppg))
    fft_elgendi = np.abs(np.fft.rfft(ppg_cleaned_elgendi))

    freqs = np.fft.rfftfreq(ppg.size, 1 / sampling_rate)

    assert np.sum(fft_raw[freqs < .5]) > np.sum(fft_elgendi[freqs < .5])
    assert np.sum(fft_raw[freqs > 8]) > np.sum(fft_elgendi[freqs > 8])


def test_ppg_findpeaks():

    sampling_rate = 500

    ppg = nk.ppg_simulate(duration=30, sampling_rate=sampling_rate,
                          heart_rate=60, frequency_modulation=.01,
                          ibi_randomness=.1, drift=1, motion_amplitude=.5,
                          powerline_amplitude=.1, burst_amplitude=1,
                          burst_number=5, random_state=42, show=True)
    ppg_cleaned_elgendi = nk.ppg_clean(ppg, sampling_rate=sampling_rate,
                                       method="elgendi")

    info_elgendi = nk.ppg_findpeaks(ppg_cleaned_elgendi,
                                    sampling_rate=sampling_rate, show=True)

    peaks = info_elgendi["PPG_Peaks"]

    assert peaks.size == 26
    assert peaks.sum() == 195399

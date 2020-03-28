# -*- coding: utf-8 -*-

import neurokit2 as nk
import numpy as np


def test_ppg_simulate():

    ppg1 = nk.ppg_simulate(duration=20, sampling_rate=500, heart_rate=70,
                           frequency_modulation=.3, ibi_randomness=.25,
                           drift_amplitude=1, motion_amplitude=.5,
                           powerline_amplitude=.1, burst_amplitude=1,
                           burst_number=5, random_state=42, show=False)
    assert ppg1.size == 20 * 500

    ppg2 = nk.ppg_simulate(duration=200, sampling_rate=1000, heart_rate=70,
                           frequency_modulation=.3, ibi_randomness=.25,
                           drift_amplitude=1, motion_amplitude=.5,
                           powerline_amplitude=.1, burst_amplitude=1,
                           burst_number=5, random_state=42, show=False)
    assert ppg2.size == 200 * 1000

    # Ensure that frequency_modulation does not affect other signal properties.
    ppg3 = nk.ppg_simulate(duration=200, sampling_rate=1000, heart_rate=70,
                           frequency_modulation=1, ibi_randomness=.25,
                           drift_amplitude=1, motion_amplitude=.5,
                           powerline_amplitude=.1, burst_amplitude=1,
                           burst_number=5, random_state=42, show=False)
    assert np.allclose((ppg2.mean() - ppg3.mean()), 0, atol=1e-2)
    assert np.allclose((ppg2.std() - ppg3.std()), 0, atol=1e-2)

    # Ensure that ibi_randomness does not affect other signal properties.
    ppg4 = nk.ppg_simulate(duration=200, sampling_rate=1000, heart_rate=70,
                           frequency_modulation=1, ibi_randomness=1,
                           drift_amplitude=1, motion_amplitude=.5,
                           powerline_amplitude=.1, burst_amplitude=1,
                           burst_number=5, random_state=42, show=False)
    assert np.allclose((ppg3.mean() - ppg4.mean()), 0, atol=1e-1)
    assert np.allclose((ppg3.std() - ppg4.std()), 0, atol=1e-1)

    # TODO: test number of peaks, test influence of different noise configurations





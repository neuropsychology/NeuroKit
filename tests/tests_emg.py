import numpy as np
import pandas as pd
import neurokit2 as nk

import scipy.stats

# =============================================================================
# EMG
# =============================================================================


def test_emg_simulate():

    emg1 = nk.emg_simulate(duration=20, length=5000, n_bursts=1)
    assert len(emg1) == 5000

    emg2 = nk.emg_simulate(duration=20, length=5000, n_bursts=15)
    assert scipy.stats.median_absolute_deviation(emg1) < scipy.stats.median_absolute_deviation(emg2)

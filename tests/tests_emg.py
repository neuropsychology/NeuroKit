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

    emg3 = nk.emg_simulate(duration=20, length=5000, n_bursts=1, duration_bursts=2.0)
#    pd.DataFrame({"EMG1":emg1, "EMG3": emg3}).plot()
    assert len(nk.signal_findpeaks(emg3, height_min=1.0)["Peaks"]) > len(nk.signal_findpeaks(emg1, height_min=1.0)["Peaks"])

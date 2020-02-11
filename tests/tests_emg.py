import numpy as np
import pandas as pd
import neurokit2 as nk

import scipy.stats
import biosppy

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

def test_emg_clean():

    sampling_rate=1000

    emg = nk.emg_simulate(duration=20, sampling_rate=sampling_rate)
    emg_cleaned = nk.emg_clean(emg, sampling_rate=sampling_rate)

    assert emg.size == emg_cleaned.size

    # Comparison to biosppy (https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/emg.py)
    original, _, _ = biosppy.tools.filter_signal(signal=emg,
                                                 ftype='butter',
                                                 band='highpass',
                                                 order=4,
                                                 frequency=100,
                                                 sampling_rate=sampling_rate)
    emg_cleaned_biosppy = nk.signal_detrend(original, order=0)
    assert np.allclose((emg_cleaned - emg_cleaned_biosppy).mean(), 0, atol=1e-6)

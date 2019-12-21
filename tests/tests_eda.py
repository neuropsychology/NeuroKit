import numpy as np
import pandas as pd
import neurokit2 as nk

# =============================================================================
# EDA
# =============================================================================

def test_eda_simulate():

    eda1 = nk.eda_simulate(duration=10, length=None, n_peaks=1)
    assert len(nk.signal_findpeaks(eda1, height_min=0.6)[0]) == 1

    eda2 = nk.eda_simulate(duration=10, length=None, n_peaks=5)
    assert len(nk.signal_findpeaks(eda2, height_min=0.6)[0]) == 5
#   pd.DataFrame({"EDA1": eda1, "EDA2": eda2}).plot()

    assert len(nk.signal_findpeaks(eda2, height_min=0.6)[0]) > len(nk.signal_findpeaks(eda1, height_min=0.6)[0])

    eda3 = nk._eda_simulate_canonical(sampling_rate=1000, length=None, time_peak=10)
    eda4 = nk._eda_simulate_canonical(sampling_rate=1000, length=None, time_peak=3)
#   pd.DataFrame({"EDA3": eda3, "EDA4": eda4}).plot()
    assert np.argmax(eda4) < np.argmax(eda3)

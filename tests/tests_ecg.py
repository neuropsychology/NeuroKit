import numpy as np
import pandas as pd
import neurokit2 as nk

# =============================================================================
# ECG
# =============================================================================


def test_ecg_simulate():

    ecg1 = nk.ecg_simulate(duration=10, length=1000)
    assert len(ecg1) == 1000

    ecg2 = nk.ecg_simulate(duration=10, length=1000, bpm=140)
    # assert np.std(ecg1) < np.std(ecg2)




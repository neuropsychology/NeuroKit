import numpy as np
import pandas as pd
import neurokit2 as nk

import scipy.stats

# =============================================================================
# ECG
# =============================================================================


def test_ecg_simulate():

    ecg1 = nk.ecg_simulate(duration=20, length=5000)
    assert len(ecg1) == 5000

    ecg2 = nk.ecg_simulate(duration=20, length=5000, bpm=500)
    pd.DataFrame({"ECG1":ecg1, "ECG2": ecg2}).plot()
#    pd.DataFrame({"ECG1":ecg1, "ECG2": ecg2}).hist()
#    assert scipy.stats.median_absolute_deviation(ecg1) < scipy.stats.median_absolute_deviation(ecg2)

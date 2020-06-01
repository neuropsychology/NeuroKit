import numpy as np
import pandas as pd

import neurokit2 as nk

sampling_rate = 1000

for heartrate in [80]:
    # Simulate signal
    ecg = nk.ecg_simulate(duration=60,
                          sampling_rate=sampling_rate,
                          heartrate=heartrate,
                          noise=0)

    # Segment
    _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
#    _, waves = nk.ecg_delineator(ecg, rpeaks=rpeaks["ECG_R_Peaks"])

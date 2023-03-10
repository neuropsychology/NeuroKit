import os.path
import numpy as np
import pandas as pd
import neurokit2 as nk
from neurokit2.ecg.ecg_interpolatepeaks import ecg_interpolatepeaks

def test_ecg_interpolatepeaks():
    # simulate data
    ecg_clean_peaks = nk.ecg_peaks(
        nk.ecg_clean(
            nk.ecg_simulate(
                duration=10,
                sampling_rate = 1000,
                heart_rate=100, 
                random_state=42)))
                
    # interpolate R-peaks
    rpeaks_interpolated = ecg_interpolatepeaks(ecg_clean_peaks)

    # test if the time series have the same number of R-peaks
    np.testing.assert_array_equal(
        pd.Series.sum(rpeaks_interpolated==1),
        pd.Series.sum(ecg_clean_peaks[0]))
    
    # test if the time series have the same length
    np.testing.assert_array_equal(
        len(rpeaks_interpolated), 
        len(ecg_clean_peaks[0]))

    # test if the interpolated time series reduced back to a binary
    # time series is equal to the output of ecg_peaks()[0]
    np.testing.assert_array_equal(
        np.array([x for x in rpeaks_interpolated==1],int),
        ecg_clean_peaks[0].to_numpy(int).flatten())

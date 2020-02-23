import pytest
import neurokit2 as nk
import pathlib
import numpy as np
import pandas as pd


def get_signal(filename=None, sampling_rate=2000):
    if filename is None:
        ecg = nk.ecg_simulate(
            duration=10, sampling_rate=sampling_rate, method="ecgsyn")
    else:
        filename = (pathlib.Path(__file__) / '../ecg_data' / filename).resolve().as_posix()
        ecg = np.array(pd.read_csv(filename))[:, 1]
    return ecg, sampling_rate


@pytest.fixture(name='test_data')
def load_ecg_signal():
    ecg, sampling_rate = get_signal('bad_500.csv', sampling_rate=500)
    yield ecg, sampling_rate


def test_find_T_peaks_correctly(test_data):
    ecg_signal, sampling_rate = test_data
    rpeaks = nk.ecg_findpeaks(ecg_signal, sampling_rate=sampling_rate, method='martinez')['ECG_R_Peaks']
    ecg_characteristics = nk.ecg_delineator(ecg_signal, rpeaks, sampling_rate, method='dwt')

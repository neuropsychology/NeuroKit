import pytest
import neurokit2 as nk
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging


SHOW_DEBUG_PLOTS = False
MAX_SIGNAL_DIFF = 0.01  # seconds


@pytest.fixture(name='test_data')
def fixture_load_ecg_data():
    """Load ecg signal and sampling rate."""
    def _get_signal(filename=None, sampling_rate=2000):
        if filename is None:
            ecg = nk.ecg_simulate(
                duration=10, sampling_rate=sampling_rate, method="ecgsyn")
        else:
            filename = (pathlib.Path(__file__) / '../ecg_data' / filename).resolve().as_posix()
            ecg = np.array(pd.read_csv(filename))[:, 1]
        return ecg, sampling_rate

    ecg, sampling_rate = _get_signal('bad_500.csv', sampling_rate=500)
    rpeaks = nk.ecg_findpeaks(ecg, sampling_rate=sampling_rate, method='martinez')['ECG_R_Peaks']
    annots = dict(
        ECG_T_Peaks=[537, 1031, 1682],
        ECG_P_Peaks=[364, 931, 1516],
        ECG_R_Onsets=[407, 973, 1559],
        ECG_R_Offsets=[456, 1021, 1605],
    )

    if SHOW_DEBUG_PLOTS:
        plt.plot(ecg)
        plt.show()

    test_data = dict(ecg=ecg, sampling_rate=sampling_rate, rpeaks=rpeaks)
    test_data.update(annots)
    yield test_data


def test_find_T_peaks(test_data):
    ecg_characteristics = nk.ecg_delineator(
        test_data['ecg'], test_data['rpeaks'], test_data['sampling_rate'], method='dwt')

    np.testing.assert_allclose(ecg_characteristics['ECG_T_Peaks'][:3],
                               test_data['ECG_T_Peaks'],
                               atol=MAX_SIGNAL_DIFF * test_data['sampling_rate'])


def test_find_P_peaks(test_data):
    ecg_characteristics = nk.ecg_delineator(
        test_data['ecg'], test_data['rpeaks'], test_data['sampling_rate'], method='dwt')

    np.testing.assert_allclose(ecg_characteristics['ECG_P_Peaks'][:3],
                               test_data['ECG_P_Peaks'],
                               atol=MAX_SIGNAL_DIFF * test_data['sampling_rate'])


def test_find_qrs_bounds(test_data):
    ecg_characteristics = nk.ecg_delineator(
        test_data['ecg'], test_data['rpeaks'], test_data['sampling_rate'], method='dwt')

    np.testing.assert_allclose(ecg_characteristics['ECG_R_Onsets'][:3],
                               test_data['ECG_R_Onsets'],
                               atol=MAX_SIGNAL_DIFF * test_data['sampling_rate'])

    np.testing.assert_allclose(ecg_characteristics['ECG_R_Offsets'][:3],
                               test_data['ECG_R_Offsets'],
                               atol=MAX_SIGNAL_DIFF * test_data['sampling_rate'])

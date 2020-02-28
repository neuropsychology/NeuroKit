import pytest
import neurokit2 as nk
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging


SHOW_DEBUG_PLOTS = False
MAX_SIGNAL_DIFF = 0.03  # seconds


@pytest.fixture(name='test_data')
def setup_load_ecg_data():
    """Load ecg signal and sampling rate."""
    def load_signal_from_disk(filename=None, sampling_rate=2000):
        if filename is None:
            ecg = nk.ecg_simulate(
                duration=10, sampling_rate=sampling_rate, method="ecgsyn")
        else:
            filename = (pathlib.Path(__file__) / '../ecg_data' / filename).resolve().as_posix()
            ecg = np.array(pd.read_csv(filename))[:, 1]
        return ecg, sampling_rate

    ecg, sampling_rate = load_signal_from_disk('good_4000.csv', sampling_rate=4000)
    annots_filename = (pathlib.Path(__file__) / '../ecg_data' / 'good_4000_annotation.csv').resolve().as_posix()
    annots = pd.read_csv(annots_filename, index_col=0, header=None).transpose()

    if SHOW_DEBUG_PLOTS:
        plt.plot(ecg)
        plt.show()

    rpeaks = nk.ecg_findpeaks(ecg, sampling_rate=sampling_rate, method='martinez')['ECG_R_Peaks']
    test_data = dict(ecg=ecg, sampling_rate=sampling_rate, rpeaks=rpeaks)
    test_data.update(annots)
    yield test_data


def helper_plot(attribute, ecg_characteristics, test_data):
    peaks = [ecg_characteristics[attribute], test_data[attribute]]
    print('0: computed\n1: data')
    ecg = test_data['ecg']
    nk.events_plot(peaks, ecg)
    plt.show()


def run_test_func(test_data):
    return nk.ecg_delineator(
        test_data['ecg'], test_data['rpeaks'], test_data['sampling_rate'], method='dwt')


def test_find_T_peaks(test_data):
    ecg_characteristics = run_test_func(test_data)
    np.testing.assert_allclose(ecg_characteristics['ECG_T_Peaks'],
                               test_data['ECG_T_Peaks'],
                               atol=MAX_SIGNAL_DIFF * test_data['sampling_rate'])


def test_find_T_onsets_offsets(test_data):
    ecg_characteristics = run_test_func(test_data)
    np.testing.assert_allclose(ecg_characteristics['ECG_T_Onsets'],
                               test_data['ECG_T_Onsets'][:-1],
                               atol=MAX_SIGNAL_DIFF * test_data['sampling_rate'])
    np.testing.assert_allclose(ecg_characteristics['ECG_T_Offsets'],
                               test_data['ECG_T_Offsets'][:-1],
                               atol=MAX_SIGNAL_DIFF * test_data['sampling_rate'])


def test_find_P_onsets_offsets(test_data):
    ecg_characteristics = run_test_func(test_data)
    np.testing.assert_allclose(ecg_characteristics['ECG_P_Onsets'],
                               test_data['ECG_P_Onsets'][:-1],
                               atol=MAX_SIGNAL_DIFF * test_data['sampling_rate'])
    np.testing.assert_allclose(ecg_characteristics['ECG_P_Offsets'],
                               test_data['ECG_P_Offsets'][:-1],
                               atol=MAX_SIGNAL_DIFF * test_data['sampling_rate'])


def test_find_P_peaks(test_data):
    ecg_characteristics = run_test_func(test_data)
    np.testing.assert_allclose(ecg_characteristics['ECG_P_Peaks'],
                               test_data['ECG_P_Peaks'][:-1],
                               atol=MAX_SIGNAL_DIFF * test_data['sampling_rate'])


def test_find_qrs_onsets(test_data):
    ecg_characteristics = run_test_func(test_data)

    np.testing.assert_allclose(ecg_characteristics['ECG_R_Onsets'],
                               test_data['ECG_R_Onsets'][:-1],
                               atol=MAX_SIGNAL_DIFF * test_data['sampling_rate'])

    np.testing.assert_allclose(ecg_characteristics['ECG_R_Offsets'],
                               test_data['ECG_R_Offsets'][1:],
                               atol=MAX_SIGNAL_DIFF * test_data['sampling_rate'])



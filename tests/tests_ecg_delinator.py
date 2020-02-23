import pytest
import neurokit2 as nk
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    yield dict(ecg=ecg, sampling_rate=sampling_rate, rpeaks=rpeaks)


def test_find_T_peaks_correctly(test_data):
    ecg_characteristics = nk.ecg_delineator(
        test_data['ecg'], test_data['rpeaks'], test_data['sampling_rate'], method='dwt')

    np.testing.assert_allclose(ecg_characteristics['ECG_T_Peaks'][:3],
                               [537, 1031, 1682], atol=MAX_SIGNAL_DIFF * test_data['sampling_rate'])

    # plt.plot(ecg_signal)
    # plt.show()


# def test_find_P_peaks_correctly(test_data):
#     ecg_signal, sampling_rate = test_data
#     rpeaks = nk.ecg_findpeaks(ecg_signal, sampling_rate=sampling_rate, method='martinez')['ECG_R_Peaks']
#     ecg_characteristics = nk.ecg_delineator(ecg_signal, rpeaks, sampling_rate, method='dwt')

#     np.testing.assert_allclose(ecg_characteristics['ECG_T_Peaks'][:3],
#                                [537, 1031, 1682], atol=MAX_SIGNAL_DIFF * sampling_rate)


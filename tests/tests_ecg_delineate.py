import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sys

import neurokit2 as nk

SHOW_DEBUG_PLOTS = False
MAX_SIGNAL_DIFF = 0.03  # seconds


@pytest.fixture(name="test_data")
def setup_load_ecg_data():
    """Load ecg signal and sampling rate."""

    def load_signal_from_disk(filename=None, sampling_rate=2000):
        if filename is None:
            ecg = nk.ecg_simulate(duration=10, sampling_rate=sampling_rate, method="ecgsyn")
        else:
            filename = (pathlib.Path(__file__) / "../ecg_data" / filename).resolve().as_posix()
            ecg = np.array(pd.read_csv(filename))[:, 1]
        return ecg, sampling_rate

    ecg, sampling_rate = load_signal_from_disk("good_4000.csv", sampling_rate=4000)
    annots_filename = (pathlib.Path(__file__) / "../ecg_data" / "good_4000_annotation.csv").resolve().as_posix()
    annots = pd.read_csv(annots_filename, index_col=0, header=None).transpose()

    if SHOW_DEBUG_PLOTS:
        plt.plot(ecg)
        plt.show()

    rpeaks = nk.ecg_findpeaks(ecg, sampling_rate=sampling_rate, method="martinez")["ECG_R_Peaks"]
    test_data = dict(ecg=ecg, sampling_rate=sampling_rate, rpeaks=rpeaks)
    test_data.update(annots)
    yield test_data


def helper_plot(attribute, ecg_characteristics, test_data):
    peaks = [ecg_characteristics[attribute], test_data[attribute]]
    print("0: computed\n1: data")
    ecg = test_data["ecg"]
    nk.events_plot(peaks, ecg)
    plt.title(attribute)
    plt.show()


def run_test_func(test_data):
    _, waves = nk.ecg_delineate(test_data["ecg"], test_data["rpeaks"], test_data["sampling_rate"], method="dwt")
    for key in waves:
        waves[key] = np.array(waves[key])
    return waves


@pytest.mark.parametrize(
    "attribute",
    [
        "ECG_P_Peaks",
        "ECG_T_Peaks",
        "ECG_T_Onsets",
        "ECG_T_Offsets",
        "ECG_P_Onsets",
        "ECG_P_Offsets",
        "ECG_R_Onsets",
        "ECG_R_Offsets",
    ],
)
@pytest.mark.skipif(sys.platform.startswith("mac"), reason="skip testing on macos")
def test_find_ecg_characteristics(attribute, test_data):
    ecg_characteristics = run_test_func(test_data)
    diff = ecg_characteristics[attribute] - test_data[attribute]
    diff = diff[diff.abs() < 0.5 * test_data["sampling_rate"]]  # remove obvious failure
    test_data[attribute] = test_data[attribute][~np.isnan(test_data[attribute])]
    diff = diff[~np.isnan(diff)]
    report = """
Difference statistics
{diff_describe}
Difference:
{diff}
""".format(
        diff_describe=diff.describe(), diff=diff
    )
    # helper_plot(attribute, ecg_characteristics, test_data)
    assert diff.std() < 0.1 * test_data["sampling_rate"], report
    assert diff.mean() < 0.1 * test_data["sampling_rate"], report

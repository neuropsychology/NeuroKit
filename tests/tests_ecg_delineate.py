import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

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
    report = f"""
Difference statistics
{diff.describe()}
Difference:
{diff}
"""
    # helper_plot(attribute, ecg_characteristics, test_data)
    assert diff.std() < 0.11 * test_data["sampling_rate"], report
    assert diff.mean() < 0.11 * test_data["sampling_rate"], report


@pytest.mark.parametrize("scale, offset", [(0.001, 0), (2, 0), (1000, 0), (1, 5.0), (250, -3.0)])
def test_ecg_delineate_dwt_is_invariant_to_signal_units(scale, offset):
    """The DWT delineation must not depend on the amplitude units (e.g., mV vs. uV) or the baseline of the ECG."""
    sampling_rate = 1000
    ecg = nk.ecg_clean(np.asarray(nk.data("ecg_1000hz")), sampling_rate=sampling_rate)
    rpeaks = nk.ecg_findpeaks(ecg, sampling_rate=sampling_rate, method="neurokit")["ECG_R_Peaks"]

    _, reference = nk.ecg_delineate(ecg, rpeaks, sampling_rate=sampling_rate, method="dwt")
    _, rescaled = nk.ecg_delineate(ecg * scale + offset, rpeaks, sampling_rate=sampling_rate, method="dwt")

    for key, expected in reference.items():
        np.testing.assert_array_equal(np.array(rescaled[key], dtype=float), np.array(expected, dtype=float), err_msg=key)


def test_dwt_normalize_window():
    from neurokit2.ecg.ecg_delineate import _dwt_normalize_window

    window = np.array([2.0, 4.0, 10.0, 6.0])  # median = 5, absolute deviations = [3, 1, 5, 1]
    expected = [0.6, 0.2, 1.0, 0.2]
    np.testing.assert_allclose(_dwt_normalize_window(window), expected)
    np.testing.assert_allclose(_dwt_normalize_window(window, weight=0.3), np.array(expected) * 0.3)
    # Invariant to scale and baseline offset, and to the polarity of the waves
    np.testing.assert_allclose(_dwt_normalize_window(window * 1000 + 7), expected)
    np.testing.assert_allclose(_dwt_normalize_window(-window), expected)
    # Degenerate windows must not produce NaN/inf
    np.testing.assert_array_equal(_dwt_normalize_window(np.full(5, 3.0)), np.zeros(5))
    assert len(_dwt_normalize_window([])) == 0

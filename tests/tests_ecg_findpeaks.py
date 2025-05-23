# -*- coding: utf-8 -*-
import os.path

import numpy as np
import pandas as pd
import pytest

# Trick to directly access internal functions for unit testing.
#
# Using neurokit2.ecg.ecg_findpeaks._ecg_findpeaks_MWA doesn't
# work because of the "from .ecg_findpeaks import ecg_findpeaks"
# statement in neurokit2/ecg/__init.__.py.
from neurokit2.ecg.ecg_findpeaks import (
    _ecg_findpeaks_MWA,
    _ecg_findpeaks_peakdetect,
    _ecg_findpeaks_hamilton,
    _ecg_findpeaks_findmethod,
)


def _read_csv_column(csv_name, column):
    csv_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "ecg_data", csv_name
    )
    csv_data = pd.read_csv(csv_path, header=None)
    return csv_data[column].to_numpy()


@pytest.mark.parametrize("method",["neurokit", "pantompkins", "nabian", "gamboa",
               "slopesumfunction", "wqrs", "hamilton", "christov",
               "engzee", "manikandan", "elgendi", "kalidas", "khamis",
               "martinez", "rodrigues", "vgraph"])
def test_ecg_findpeaks_all_methods_handle_empty_input(method):
    method_func = _ecg_findpeaks_findmethod(method)
    # The test here is implicit: no exceptions means that it passed,
    # even if the output is nonsense.
    _ = method_func(np.zeros(12*240), sampling_rate=240)


def test_ecg_findpeaks_MWA():
    np.testing.assert_array_equal(
        _ecg_findpeaks_MWA(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float), 3),
        [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8],
    )


# This test case is intentionally a "change aversion" test that simply
# verifies that the output of the _ecg_findpeaks_peakdetect function
# on two different test datasets remains unchanged.
#
# Most notably the assertions here don't necessarily document the
# "correct" output of the function, just what the output used to be earlier.
# Potential bug fixes could legitimately require updates to this test case.
#
# Instead the main purpose of this test case is to give extra confidence
# that optimizations or other refactorings won't accidentally introduce
# new bugs into the function.
def test_ecg_findpeaks_peakdetect():
    good_4000 = _read_csv_column("good_4000.csv", 1)
    expected_good_4000_peaks = _read_csv_column(
        "expected_ecg_findpeaks_peakdetect_good_4000.csv", 0
    )
    np.testing.assert_array_equal(
        _ecg_findpeaks_peakdetect(good_4000, sampling_rate=4000),
        expected_good_4000_peaks,
    )

    bad_500 = _read_csv_column("bad_500.csv", 1)
    expected_bad_500_peaks = _read_csv_column(
        "expected_ecg_findpeaks_peakdetect_bad_500.csv", 0
    )
    np.testing.assert_array_equal(
        _ecg_findpeaks_peakdetect(bad_500, sampling_rate=500), expected_bad_500_peaks
    )


def test_ecg_findpeaks_hamilton():
    good_4000 = _read_csv_column("good_4000.csv", 1)
    expected_good_4000_peaks = _read_csv_column(
        "expected_ecg_findpeaks_hamilton_good_4000.csv", 0
    )
    np.testing.assert_array_equal(
        _ecg_findpeaks_hamilton(good_4000, sampling_rate=4000), expected_good_4000_peaks
    )

    bad_500 = _read_csv_column("bad_500.csv", 1)
    expected_bad_500_peaks = _read_csv_column(
        "expected_ecg_findpeaks_hamilton_bad_500.csv", 0
    )
    np.testing.assert_array_equal(
        _ecg_findpeaks_hamilton(bad_500, sampling_rate=500), expected_bad_500_peaks
    )

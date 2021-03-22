# -*- coding: utf-8 -*-
import numpy as np
import pytest

# Trick to directly access the internal function.
# Using neurokit2.ecg.ecg_findpeaks._ecg_findpeaks_MWA doesn't
# work because of the "from .ecg_findpeaks import ecg_findpeaks"
# statement in neurokit2/ecg/__init.__.py.
from neurokit2.ecg.ecg_findpeaks import _ecg_findpeaks_MWA


def test_ecg_findpeaks_MWA():
    np.testing.assert_array_equal(
        _ecg_findpeaks_MWA(
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float),
            3),
        [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])

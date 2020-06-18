import os

import numpy as np

import neurokit2 as nk

path_data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# =============================================================================
# Data
# =============================================================================


def test_read_acqknowledge():

    df, sampling_rate = nk.read_acqknowledge(os.path.join(path_data, "acqnowledge.acq"), sampling_rate=2000)
    assert sampling_rate == 2000

    df, sampling_rate = nk.read_acqknowledge(os.path.join(path_data, "acqnowledge.acq"), sampling_rate="max")
    assert sampling_rate == 4000


def test_data():

    dataset = "bio_eventrelated_100hz"

    data = nk.data(dataset)
    assert len(data.columns) == 4
    assert data.size == 15000 * 4
    assert all(elem in ["ECG", "EDA", "Photosensor", "RSP"] for elem in np.array(data.columns.values, dtype=str))

    dataset2 = "bio_eventrelated_100hz.csv"

    data2 = nk.data(dataset2)
    assert len(data.columns) == len(data2.columns)
    assert data2.size == data.size
    assert all(elem in np.array(data.columns.values, dtype=str) for elem in np.array(data2.columns.values, dtype=str))

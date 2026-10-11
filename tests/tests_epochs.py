import numpy as np
import pytest

import neurokit2 as nk


def test_epochs_create():
    # Get data
    data = nk.data("bio_eventrelated_100hz")

    # Find events
    events = nk.events_find(
        data["Photosensor"], threshold_keep="below", event_conditions=["Negative", "Neutral", "Neutral", "Negative"]
    )

    # Create epochs
    epochs_1 = nk.epochs_create(data, events, sampling_rate=100, epochs_start=-0.5, epochs_end=3)
    epochs_2 = nk.epochs_create(data, events, sampling_rate=100, epochs_start=-0.5, epochs_end=1)

    # Test lengths and column names
    assert len(epochs_1) == 4
    columns = ["ECG", "EDA", "Photosensor", "RSP", "Index", "Label", "Condition"]
    assert all(elem in columns for elem in np.array(epochs_1["1"].columns.values, dtype=str))

    # Test corresponding event features in epochs
    condition_names = []
    for i in epochs_1:
        cond = np.unique(epochs_1[i].Condition)[0]
        condition_names.append(cond)
        assert events["onset"][int(i) - 1] in np.array(epochs_1[i].Index)
    assert events["condition"] == condition_names

    # Test full vs subsetted epochs
    for i, j in zip(epochs_1, epochs_2):
        epoch_full = epochs_2[str(j)]
        epoch_subset = epochs_1[str(i)].loc[-0.5:1]

        assert len(epoch_full) == len(epoch_subset)
        for col in epoch_full.columns:
            assert all(np.array(epoch_subset[col]) == np.array(epoch_full[col]))


@pytest.mark.parametrize("starts", [[-0.5, -1], [0, 0.2], [-0.5, 0, 0.5]])
def test_epochs_create_baseline_per_epoch(starts):
    data = np.arange(1000.0) ** 2
    events = [200, 400, 600][: len(starts)]
    raw = nk.epochs_create(data, events, sampling_rate=100, epochs_start=starts, epochs_end=1)
    corrected = nk.epochs_create(data, events, sampling_rate=100, epochs_start=starts, epochs_end=1, baseline_correction=True)

    for epoch, start in zip(raw, starts):
        signal = raw[epoch]["Signal"]
        baseline = signal.loc[:0] if start < 0 else signal.iloc[:1]
        expected = signal - baseline.mean()
        np.testing.assert_allclose(corrected[epoch]["Signal"], expected)
        np.testing.assert_array_equal(corrected[epoch].index, expected.index)
        np.testing.assert_array_equal(corrected[epoch]["Label"], raw[epoch]["Label"])

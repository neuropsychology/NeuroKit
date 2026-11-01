import numpy as np
import pytest

import neurokit2 as nk


@pytest.mark.parametrize("sampling_rate", [100, 128])
@pytest.mark.parametrize("duration", [1, 2, 2.5])
@pytest.mark.parametrize("start", [-0.5, 0, 0.5])
def test_epochs_create_complete_blocks(sampling_rate, duration, start):
    signal = np.arange(int(duration * sampling_rate), dtype=float)
    epochs = nk.epochs_create(signal, sampling_rate=sampling_rate, epochs_start=start, epochs_end=start + 1)

    assert len(epochs) == int(duration)
    for number, epoch in enumerate(epochs.values()):
        first = number * sampling_rate
        expected = signal[first : first + sampling_rate]
        np.testing.assert_array_equal(epoch["Signal"], expected)
        np.testing.assert_array_equal(epoch["Index"], np.arange(first, first + sampling_rate))
        assert epoch.index[0] == start


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

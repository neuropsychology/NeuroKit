import biosppy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.stats

import neurokit2 as nk

# =============================================================================
# EMG
# =============================================================================


def test_emg_simulate():
    emg1 = nk.emg_simulate(duration=20, length=5000, burst_number=1)
    assert len(emg1) == 5000

    emg2 = nk.emg_simulate(duration=20, length=5000, burst_number=15)
    assert scipy.stats.median_abs_deviation(emg1) < scipy.stats.median_abs_deviation(
        emg2
    )

    emg3 = nk.emg_simulate(duration=20, length=5000, burst_number=1, burst_duration=2.0)
    #    pd.DataFrame({"EMG1":emg1, "EMG3": emg3}).plot()
    assert len(nk.signal_findpeaks(emg3, height_min=1.0)["Peaks"]) > len(
        nk.signal_findpeaks(emg1, height_min=1.0)["Peaks"]
    )


def test_emg_activation():
    emg = nk.emg_simulate(duration=10, burst_number=3)
    cleaned = nk.emg_clean(emg)
    emg_amplitude = nk.emg_amplitude(cleaned)

    activity_signal, info = nk.emg_activation(emg_amplitude)

    assert set(activity_signal.columns.to_list()) == set(list(info.keys()))
    assert len(info["EMG_Onsets"]) == len(info["EMG_Offsets"])
    for i, j in zip(info["EMG_Onsets"], info["EMG_Offsets"]):
        assert i < j


def test_emg_clean():
    sampling_rate = 1000

    emg = nk.emg_simulate(duration=20, sampling_rate=sampling_rate)
    emg_cleaned = nk.emg_clean(emg, sampling_rate=sampling_rate)

    assert emg.size == emg_cleaned.size

    # Comparison to biosppy (https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/emg.py)
    original, _, _ = biosppy.tools.filter_signal(
        signal=emg,
        ftype="butter",
        band="highpass",
        order=4,
        frequency=100,
        sampling_rate=sampling_rate,
    )
    emg_cleaned_biosppy = nk.signal_detrend(original, order=0)
    assert np.allclose((emg_cleaned - emg_cleaned_biosppy).mean(), 0, atol=1e-6)


def test_emg_plot():
    sampling_rate = 1000

    emg = nk.emg_simulate(duration=10, sampling_rate=1000, burst_number=3)
    emg_summary, info = nk.emg_process(emg, sampling_rate=sampling_rate)

    # Plot data over samples.
    nk.emg_plot(emg_summary, info)
    fig = plt.gcf()
    assert len(fig.axes) == 2
    titles = ["Raw and Cleaned Signal", "Muscle Activation"]
    for ax, title in zip(fig.get_axes(), titles):
        assert ax.get_title() == title
    assert fig.get_axes()[1].get_xlabel() == "Time (seconds)"
    np.testing.assert_array_equal(fig.axes[0].get_xticks(), fig.axes[1].get_xticks())
    plt.close(fig)


def test_emg_eventrelated():
    emg = nk.emg_simulate(duration=20, sampling_rate=1000, burst_number=3)
    emg_signals, info = nk.emg_process(emg, sampling_rate=1000)
    epochs = nk.epochs_create(
        emg_signals,
        events=[3000, 6000, 9000],
        sampling_rate=1000,
        epochs_start=-0.1,
        epochs_end=1.9,
    )
    emg_eventrelated = nk.emg_eventrelated(epochs)

    # Test amplitude features
    no_activation = np.where(emg_eventrelated["EMG_Activation"] == 0)[0][0]
    assert int(pd.DataFrame(emg_eventrelated.values[no_activation]).isna().sum()) == 5

    assert np.all(
        np.nansum(np.array(emg_eventrelated["EMG_Amplitude_Mean"]))
        < np.nansum(np.array(emg_eventrelated["EMG_Amplitude_Max"]))
    )

    assert len(emg_eventrelated["Label"]) == 3

    # Test warning on missing columns
    with pytest.warns(
        nk.misc.NeuroKitWarning, match=r".*does not have an `EMG_Onsets`.*"
    ):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["EMG_Onsets"]
        nk.emg_eventrelated({**epochs, first_epoch_key: first_epoch_copy})

    with pytest.warns(
        nk.misc.NeuroKitWarning, match=r".*does not have an `EMG_Activity`.*"
    ):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["EMG_Activity"]
        nk.emg_eventrelated({**epochs, first_epoch_key: first_epoch_copy})

    with pytest.warns(
        nk.misc.NeuroKitWarning, match=r".*does not have an.*`EMG_Amplitude`.*"
    ):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["EMG_Amplitude"]
        nk.emg_eventrelated({**epochs, first_epoch_key: first_epoch_copy})


def test_emg_intervalrelated():
    emg = nk.emg_simulate(duration=40, sampling_rate=1000, burst_number=3)
    emg_signals, info = nk.emg_process(emg, sampling_rate=1000)
    columns = ["EMG_Activation_N", "EMG_Amplitude_Mean"]

    # Test with signal dataframe
    features_df = nk.emg_intervalrelated(emg_signals)

    assert all(
        elem in columns for elem in np.array(features_df.columns.values, dtype=str)
    )
    assert features_df.shape[0] == 1  # Number of rows

    # Test with dict
    columns.append("Label")
    epochs = nk.epochs_create(
        emg_signals, events=[0, 20000], sampling_rate=1000, epochs_end=20
    )
    features_dict = nk.emg_intervalrelated(epochs)

    assert all(
        elem in columns for elem in np.array(features_dict.columns.values, dtype=str)
    )
    assert features_dict.shape[0] == 2  # Number of rows


@pytest.mark.parametrize(
    "method_cleaning, method_activation, threshold",
    [
        ("none", "threshold", "default"),
        ("biosppy", "pelt", 0.5),
        ("biosppy", "mixture", 0.05),
        ("biosppy", "biosppy", "default"),
        ("biosppy", "silva", "default"),
    ],
)
def test_emg_report(tmp_path, method_cleaning, method_activation, threshold):
    sampling_rate = 250

    emg = nk.emg_simulate(
        duration=30,
        sampling_rate=sampling_rate,
        random_state=0,
    )

    d = tmp_path / "sub"
    d.mkdir()
    p = d / "myreport.html"

    signals, _ = nk.emg_process(
        emg,
        sampling_rate=sampling_rate,
        report=str(p),
        method_cleaning=method_cleaning,
        method_activation=method_activation,
        threshold=threshold,
    )
    assert p.is_file()
    assert "EMG_Activity" in signals.columns

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

import neurokit2 as nk


def test_eog_clean():
    # test with exported csv
    eog_signal = nk.data("eog_200hz")["vEOG"]
    eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=200)
    assert eog_cleaned.size == eog_signal.size

    # test with mne.io.Raw
    raw = mne.io.read_raw_fif(
        str(mne.datasets.sample.data_path()) + "/MEG/sample/sample_audvis_raw.fif",
        preload=True,
    )
    sampling_rate = raw.info["sfreq"]

    eog_channels = nk.mne_channel_extract(raw, what="EOG", name="EOG").values
    eog_cleaned = nk.eog_clean(eog_channels, sampling_rate, method="agarwal2019")
    assert eog_cleaned.size == eog_channels.size

    # compare with mne filter
    eog_cleaned_mne = nk.eog_clean(eog_channels, sampling_rate, method="mne")
    mne_clean = mne.filter.filter_data(
        eog_channels,
        sfreq=sampling_rate,
        l_freq=1,
        h_freq=10,
        filter_length="10s",
        l_trans_bandwidth=0.5,
        h_trans_bandwidth=0.5,
        phase="zero-double",
        fir_window="hann",
        fir_design="firwin2",
        verbose=False,
    )
    assert np.allclose((eog_cleaned_mne - mne_clean).mean(), 0)


def test_eog_findpeaks():
    eog_signal = nk.data("eog_100hz")
    eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=100)

    # Test with NeuroKit
    nk_peaks = nk.eog_findpeaks(
        eog_cleaned, sampling_rate=100, method="neurokit", threshold=0.33, show=False
    )
    assert nk_peaks.size == 19

    # Test with MNE
    mne_peaks = nk.eog_findpeaks(eog_cleaned, method="mne")
    assert mne_peaks.size == 44

    # Test with brainstorm
    brainstorm_peaks = nk.eog_findpeaks(eog_cleaned, method="brainstorm")
    assert brainstorm_peaks.size == 28

    blinker_peaks = nk.eog_findpeaks(eog_cleaned, method="blinker", sampling_rate=100)
    assert blinker_peaks.size == 14


def test_eog_process():
    eog_signal = nk.data("eog_200hz")["vEOG"]
    signals, info = nk.eog_process(eog_signal, sampling_rate=200)

    # Extract blinks, test across dataframe and dict
    blinks = np.where(signals["EOG_Blinks"] == 1)[0]
    assert np.all(blinks == info["EOG_Blinks"])


def test_eog_plot():
    eog_signal = nk.data("eog_100hz")
    signals, info = nk.eog_process(eog_signal, sampling_rate=100)

    # Plot
    nk.eog_plot(signals, info)
    fig = plt.gcf()
    assert len(fig.axes) == 3

    titles = ["Raw and Cleaned Signal", "Blink Rate", "Individual Blinks"]
    legends = [["Raw", "Cleaned", "Blinks"], ["Rate", "Mean"], ["Median"]]
    ylabels = ["Amplitude (mV)", "Blinks per minute"]

    for ax, title, legend, ylabel in zip(fig.get_axes(), titles, legends, ylabels):
        assert ax.get_title() == title
        subplot = ax.get_legend_handles_labels()
        assert subplot[1] == legend
        assert ax.get_ylabel() == ylabel

    assert fig.get_axes()[1].get_xlabel() == "Time (seconds)"
    np.testing.assert_array_equal(fig.axes[0].get_xticks(), fig.axes[1].get_xticks())
    plt.close(fig)

    with pytest.raises(ValueError, match=r"NeuroKit error: eog_plot.*"):
        nk.eog_plot(None)


def test_eog_eventrelated():
    eog = nk.data("eog_200hz")["vEOG"]
    eog_signals, info = nk.eog_process(eog, sampling_rate=200)
    epochs = nk.epochs_create(
        eog_signals, events=[5000, 10000, 15000], epochs_start=-0.1, epochs_end=1.9
    )
    eog_eventrelated = nk.eog_eventrelated(epochs)

    # Test rate features
    assert np.all(
        np.array(eog_eventrelated["EOG_Rate_Min"])
        < np.array(eog_eventrelated["EOG_Rate_Mean"])
    )

    assert np.all(
        np.array(eog_eventrelated["EOG_Rate_Mean"])
        < np.array(eog_eventrelated["EOG_Rate_Max"])
    )

    # Test blink presence
    assert np.all(
        np.array(eog_eventrelated["EOG_Blinks_Presence"]) == np.array([1, 0, 0])
    )

    # Test warning on missing columns
    with pytest.warns(
        nk.misc.NeuroKitWarning, match=r".*does not have an `EOG_Blinks`.*"
    ):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["EOG_Blinks"]
        nk.eog_eventrelated({**epochs, first_epoch_key: first_epoch_copy})

    with pytest.warns(
        nk.misc.NeuroKitWarning, match=r".*does not have an `EOG_Rate`.*"
    ):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["EOG_Rate"]
        nk.eog_eventrelated({**epochs, first_epoch_key: first_epoch_copy})


def test_eog_intervalrelated():
    eog = nk.data("eog_200hz")["vEOG"]
    eog_signals, info = nk.eog_process(eog, sampling_rate=200)

    columns = ["EOG_Peaks_N", "EOG_Rate_Mean"]

    # Test with signal dataframe
    features = nk.eog_intervalrelated(eog_signals)

    assert all(elem in np.array(features.columns.values, dtype=str) for elem in columns)
    assert features.shape[0] == 1  # Number of rows

    # Test with dict
    columns.append("Label")
    epochs = nk.epochs_create(
        eog_signals, events=[5000, 10000, 15000], epochs_start=-0.1, epochs_end=1.9
    )
    epochs_dict = nk.eog_intervalrelated(epochs)

    assert all(
        elem in columns for elem in np.array(epochs_dict.columns.values, dtype=str)
    )
    assert epochs_dict.shape[0] == len(epochs)  # Number of rows

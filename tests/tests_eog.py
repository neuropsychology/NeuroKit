# -*- coding: utf-8 -*-
import numpy as np
import mne
import matplotlib.pyplot as plt

import neurokit2 as nk


def test_eog_clean():

    # test with exported csv
    eog_signal = nk.data("eog_200hz")["vEOG"]
    eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=200)
    assert eog_cleaned.size == eog_signal.size

    # test with mne.io.Raw
    raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() + "/MEG/sample/sample_audvis_raw.fif", preload=True)
    sampling_rate = raw.info["sfreq"]

    eog_channels = nk.mne_channel_extract(raw, what="EOG", name="EOG")
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

    eog_signal = nk.data('eog_100hz')
    eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=100)

    # Test with Neurokit
    nk_peaks = nk.eog_findpeaks(eog_cleaned, sampling_rate=100, method="neurokit", threshold=0.33, show=False)
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

    eog_signal = nk.data("eog_200hz")["vEOG"]
    signals, info = nk.eog_process(eog_signal, sampling_rate=200)

    # Plot
    nk.eog_plot(signals)
    fig = plt.gcf()
    assert len(fig.axes) == 2

    titles = ["Raw and Cleaned Signal", "Blink Rate"]
    legends = [["Raw", "Cleaned", "Blinks"], ["Rate", "Mean"]]
    ylabels = ["Amplitude (mV)", "Blinks per minute"]

    for (ax, title, legend, ylabel) in zip(fig.get_axes(), titles, legends, ylabels):
        assert ax.get_title() == title
        subplot = ax.get_legend_handles_labels()
        assert subplot[1] == legend
        assert ax.get_ylabel() == ylabel

    assert fig.get_axes()[1].get_xlabel() == "Samples"
    np.testing.assert_array_equal(fig.axes[0].get_xticks(), fig.axes[1].get_xticks())
    plt.close(fig)

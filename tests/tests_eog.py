# -*- coding: utf-8 -*-
import numpy as np
import mne

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


def test_eog_process():

    eog_signal = nk.data("eog_200hz")["vEOG"]
    signals, info = nk.eog_process(eog_signal, sampling_rate=200)

    # Extract blinks, test across dataframe and dict
    blinks = np.where(signals["EOG_Blinks"] == 1)[0]
    assert np.all(blinks == info["EOG_Blinks"])

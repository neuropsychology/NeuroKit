# -*- coding: utf-8 -*-
import mne

import neurokit2 as nk

# =============================================================================
# EEG
# =============================================================================


def test_eog_clean():

    # test with exported csv
    eog_signal = nk.data('eog_200hz')["vEOG"]
    eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=100)
    assert len(eog_cleaned) == len(eog_signal)

    # test with mne.io.Raw
    raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() +
                              '/MEG/sample/sample_audvis_raw.fif', preload=True)
    eog_channels = nk.mne_channel_extract(raw, what='EOG', name='EOG')
    eog_cleaned2 = nk.eog_clean(eog_channels, sampling_rate=raw.info['sfreq'])
    assert len(eog_cleaned2) == len(eog_channels)

# eog_process needs some fixing first

import numpy as np
import pandas as pd
import neurokit2 as nk

import mne


# =============================================================================
# EEG
# =============================================================================


def test_eeg_add_channel():


    raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_raw.fif', preload=True)

    # len(channel) > len(raw)
    ecg1 = nk.ecg_simulate(length=170000)

    # sync_index_raw > sync_index_channel
    raw1 = nk.eeg_add_channel(raw.copy(), ecg1, channel_type="ecg", sync_index_raw=100, sync_index_channel=0)
    df1 = raw1.to_data_frame()

    # test if the column of channel is added
    assert len(df1.columns) == 377

    # test if the NaN is appended properly to the added channel to account for difference in distance between two signals
    sync_index_raw=100
    sync_index_channel=0
    for i in df1["Added_Channel"].head(abs(sync_index_channel - sync_index_raw)):
        assert np.isnan(i)
    assert np.isfinite(df1["Added_Channel"].iloc[abs(sync_index_channel - sync_index_raw)])





    # len(channel) < len(raw)
    ecg2 = nk.ecg_simulate(length=166790)

    # sync_index_raw < sync_index_channel
    raw2 = nk.eeg_add_channel(raw.copy(), ecg2, channel_type="ecg", sync_index_raw=0, sync_index_channel=100)
    df2 = raw2.to_data_frame()

    # test if the column of channel is added
    assert len(df2.columns) == 377

    # test if the NaN is appended properly to the added channel to account for difference in distance between two signals + difference in length
    sync_index_raw=0
    sync_index_channel=100
    for i in df2["Added_Channel"].tail(abs(sync_index_channel - sync_index_raw) + (len(raw) - len(ecg2))):
        assert np.isnan(i)
    assert np.isfinite(df2["Added_Channel"].iloc[-abs(sync_index_channel - sync_index_raw) - (len(raw) - len(ecg2)) - 1])
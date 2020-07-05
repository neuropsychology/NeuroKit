import mne
import numpy as np

import neurokit2 as nk

# =============================================================================
# EEG
# =============================================================================


def test_eeg_add_channel():

    raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() + "/MEG/sample/sample_audvis_raw.fif", preload=True)

    # len(channel) > len(raw)
    ecg1 = nk.ecg_simulate(length=170000)

    # sync_index_raw > sync_index_channel
    raw1 = nk.mne_channel_add(raw.copy(), ecg1, channel_type="ecg", sync_index_raw=100, sync_index_channel=0)
    df1 = raw1.to_data_frame()

    # test if the column of channel is added
    assert len(df1.columns) == 378

    # test if the NaN is appended properly to the added channel to account for difference in distance between two signals
    sync_index_raw = 100
    sync_index_channel = 0
    for i in df1["Added_Channel"].head(abs(sync_index_channel - sync_index_raw)):
        assert np.isnan(i)
    assert np.isfinite(df1["Added_Channel"].iloc[abs(sync_index_channel - sync_index_raw)])

    # len(channel) < len(raw)
    ecg2 = nk.ecg_simulate(length=166790)

    # sync_index_raw < sync_index_channel
    raw2 = nk.mne_channel_add(raw.copy(), ecg2, channel_type="ecg", sync_index_raw=0, sync_index_channel=100)
    df2 = raw2.to_data_frame()

    # test if the column of channel is added
    assert len(df2.columns) == 378

    # test if the NaN is appended properly to the added channel to account for difference in distance between two signals + difference in length
    sync_index_raw = 0
    sync_index_channel = 100
    for i in df2["Added_Channel"].tail(abs(sync_index_channel - sync_index_raw) + (len(raw) - len(ecg2))):
        assert np.isnan(i)
    assert np.isfinite(
        df2["Added_Channel"].iloc[-abs(sync_index_channel - sync_index_raw) - (len(raw) - len(ecg2)) - 1]
    )


def test_mne_channel_extract():

    raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() + "/MEG/sample/sample_audvis_raw.fif", preload=True)

    # Extract 1 channel
    what = "EEG 053"

    raw_channel = nk.mne_channel_extract(raw, what)
    assert raw_channel.what == what

    # Extract more than 1 channel
    what2 = ["EEG 053", "EEG 054", "EEG 055"]

    raw_channel2 = nk.mne_channel_extract(raw, what2)
    assert len(raw_channel2.columns) == 3
    assert all(elem in what2 for elem in np.array(raw_channel2.columns.values, dtype=str))

    # Extract a category of channels
    what3 = "EEG"

    raw_channels = nk.mne_channel_extract(raw, what3)
    assert len(raw_channels.columns) == 60

    raw_eeg_names = [x for x in raw.info["ch_names"] if what3 in x]
    assert raw_eeg_names == list(raw_channels.columns.values)


def test_mne_to_df():

    raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw.fif')
    assert len(nk.mne_to_df(raw)) == 41700


    events = mne.read_events(mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif')
    event_id = {'audio/left': 1, 'audio/right': 2,
                'visual/left': 3, 'visual/right': 4}

    # Create epochs (100 ms baseline + 500 ms)
    epochs = mne.Epochs(raw,
                        events,
                        event_id,
                        tmin=-0.1,
                        tmax=0.5,
                        picks='eeg',
                        preload=True,
                        detrend=0,
                        baseline=(None, 0))
    assert len(nk.mne_to_df(epochs)) == 26208

    evoked = [epochs[name].average() for name in ('audio', 'visual')]
    assert len(nk.mne_to_df(evoked)) == 182

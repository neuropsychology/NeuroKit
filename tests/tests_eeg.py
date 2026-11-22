import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

import neurokit2 as nk


# =============================================================================
# EEG
# =============================================================================


def test_eeg_add_channel():
    raw = mne.io.read_raw_fif(str(mne.datasets.sample.data_path()) + "/MEG/sample/sample_audvis_raw.fif", preload=True)

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

    # test if the NaN is appended properly to the added channel to account for difference
    # in distance between two signals + difference in length
    sync_index_raw = 0
    sync_index_channel = 100
    for i in df2["Added_Channel"].tail(abs(sync_index_channel - sync_index_raw) + (len(raw) - len(ecg2))):
        assert np.isnan(i)
    assert np.isfinite(df2["Added_Channel"].iloc[-abs(sync_index_channel - sync_index_raw) - (len(raw) - len(ecg2)) - 1])


def test_eeg_badchannels():
    # test with numpy array input
    np.random.seed(33)
    eeg_data = np.random.randn(5, 1000)  # 5 channels, 1000 time points
    # add outliers to make one channel "bad"
    eeg_data[2, :] *= 10  # channel 2 will have much higher amplitude

    bads, results = nk.eeg_badchannels(eeg_data, bad_threshold=0.3, distance_threshold=0.95, show=False)
    assert len(results) == 5

    expected_columns = [
        "SD",
        "Mean",
        "MAD",
        "Median",
        "Skewness",
        "Kurtosis",
        "Amplitude",
        "CI_low",
        "CI_high",
        "n_ZeroCrossings",
        "Bad",
    ]
    for col in expected_columns:
        assert col in results.columns

    assert bads == [2]

    # test stats calc with longer data that works with hdi
    simple_data = np.random.randn(2, 100)  # enough points for hdi
    prev_fignums = len(plt.get_fignums())
    bads_simple, results_simple = nk.eeg_badchannels(simple_data, bad_threshold=0.8, distance_threshold=0.99)

    # No bads
    assert len(bads_simple) == 0

    # check all stats computed
    assert not np.isnan(results_simple.loc[0, "Mean"])
    assert not np.isnan(results_simple.loc[0, "SD"])
    assert results_simple.loc[0, "Amplitude"] > 0

    # Test mne Raw object input and test the show command
    mne_info = mne.create_info(ch_names=["0", "1"], sfreq=100, ch_types=["eeg"] * 2)

    mne_raw = mne.io.RawArray(data=simple_data, info=mne_info)

    bads_mne, results_mne = nk.eeg_badchannels(mne_raw, show=True)

    # Check if a figure opened
    assert len(plt.get_fignums()) == prev_fignums + 1
    plt.close("all")

    # Check if results are the same as above
    assert bads_mne == bads_simple
    assert results_mne.equals(results_simple)

    # Test Pandas DataFrame
    df_simple = pd.DataFrame(simple_data.T, columns=["0", "1"])

    bads_pandas, results_pandas = nk.eeg_badchannels(df_simple)

    # Check if results are the same as above
    assert bads_pandas == bads_simple
    assert results_pandas.equals(results_simple)

    # test bad channel detection logic
    extreme_data = np.random.randn(2, 100)
    extreme_data[1, :] *= 50  # make channel 1 extreme
    bads_extreme, results_extreme = nk.eeg_badchannels(extreme_data, bad_threshold=0.1, distance_threshold=0.7)

    # All bads
    assert len(bads_extreme) == 2

    # check bad scores calculated
    assert "Bad" in results_extreme.columns
    assert not np.isnan(results_extreme.loc[1, "Bad"])
    assert results_extreme.loc[1, "SD"] > results_extreme.loc[0, "SD"]

    # test return types
    assert isinstance(bads_simple, list)
    assert isinstance(results_simple, pd.DataFrame)
    assert results_simple.index.name == "Channel"


def test_mne_channel_extract():
    raw = mne.io.read_raw_fif(str(mne.datasets.sample.data_path()) + "/MEG/sample/sample_audvis_raw.fif", preload=True)

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
    raw = mne.io.read_raw_fif(str(mne.datasets.sample.data_path()) + "/MEG/sample/sample_audvis_filt-0-40_raw.fif")
    assert len(nk.mne_to_df(raw)) == 41700

    events = mne.read_events(str(mne.datasets.sample.data_path()) + "/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif")
    event_id = {"audio/left": 1, "audio/right": 2, "visual/left": 3, "visual/right": 4}

    # Create epochs (100 ms baseline + 500 ms)
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=-0.1,
        tmax=0.5,
        picks="eeg",
        preload=True,
        detrend=0,
        baseline=(None, 0),
    )
    assert len(nk.mne_to_df(epochs)) == 26208

    evoked = [epochs[name].average() for name in ("audio", "visual")]
    assert len(nk.mne_to_df(evoked)) == 182


def test_eeg_gfp_and_diss():
    import pytest

    rng = np.random.default_rng(42)
    # Channel 0 has 100x scale of channels 1..3; standardize_eeg=True must standardize across time (axis=1)
    eeg = rng.standard_normal((4, 200))
    eeg[0] *= 100.0
    gfp_z = nk.eeg_gfp(eeg, method="l2", standardize_eeg=True)
    expected_z = np.std(nk.standardize(eeg.T).T, axis=0, ddof=0)
    np.testing.assert_allclose(gfp_z, expected_z)
    assert np.std(gfp_z) > 0.05  # Not collapsed to a constant flat line!

    # DataFrame (times, channels) and MNE EpochsArray (3D) must match 2D (channels, times)
    df = pd.DataFrame(eeg.T, columns=["Fp1", "Fp2", "Cz", "Pz"])
    np.testing.assert_allclose(nk.eeg_gfp(df, method="l1"), nk.eeg_gfp(eeg, method="l1"))

    info = mne.create_info(["Fp1", "Fp2", "Cz", "Pz"], sfreq=100, ch_types="eeg")
    epochs = mne.EpochsArray(np.stack([eeg[:, :100], eeg[:, 100:]], axis=0), info, verbose=False)
    np.testing.assert_allclose(
        nk.eeg_gfp(epochs, method="l2", smooth=0.05), nk.eeg_gfp(eeg, sampling_rate=100, method="l2", smooth=0.05)
    )
    assert np.all(nk.eeg_gfp(np.zeros((4, 20)), normalize=True) == 0.0)
    with pytest.raises(ValueError, match="not recognized"):
        nk.eeg_gfp(eeg, method="l3")

    # DISS: identical topographies -> 0, inverted L2-normalized topographies -> 2, reference-independent, zero-GFP safe
    map_a = np.array([1.0, -1.0, 1.0, -1.0])
    eeg_flip = np.column_stack([map_a, map_a, -map_a, np.zeros(4)])
    gfp_l2 = nk.eeg_gfp(eeg_flip, method="l2")
    diss = nk.eeg_diss(eeg_flip, gfp=gfp_l2)
    np.testing.assert_allclose(diss[:3], [0.0, 0.0, 2.0])
    assert np.isfinite(diss[3])
    # Reference-independence: adding common-mode offset across channels must not change DISS
    np.testing.assert_allclose(nk.eeg_diss(eeg + 50.0, method="l2"), nk.eeg_diss(eeg, method="l2"))
    np.testing.assert_allclose(nk.eeg_diss(df, method="l2"), nk.eeg_diss(eeg, method="l2"))
    np.testing.assert_allclose(nk.eeg_diss(epochs, method="l2"), nk.eeg_diss(eeg, method="l2"))


def test_eeg_power_badchannels_and_mne_crop():
    import pytest

    rng = np.random.default_rng(7)
    t = np.arange(500) / 100.0
    eeg = np.vstack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 20 * t), 0.1 * rng.standard_normal(500)])

    # Repeated default calls must preserve band column names (no mutable default mutation)
    pwr1 = nk.eeg_power(eeg, sampling_rate=100)
    pwr2 = nk.eeg_power(eeg, sampling_rate=100)
    assert list(pwr1.columns) == ["Channel", "Gamma", "Beta", "Alpha", "Theta", "Delta"]
    pd.testing.assert_frame_equal(pwr1, pwr2)

    # Caller's list must not be mutated; spaced band names ("Beta 1") and MNE EpochsArray must work
    caller_bands = ["Alpha", "Beta 1", (18, 22)]
    pwr_custom = nk.eeg_power(eeg, sampling_rate=100, frequency_band=caller_bands, normalize=False)
    assert caller_bands == ["Alpha", "Beta 1", (18, 22)]
    assert list(pwr_custom.columns) == ["Channel", "Alpha", "Beta 1", "Hz_18_22"]
    assert pwr_custom.loc[0, "Alpha"] > pwr_custom.loc[1, "Alpha"]
    assert pwr_custom.loc[1, "Hz_18_22"] > pwr_custom.loc[0, "Hz_18_22"]

    info = mne.create_info(["Fz", "Cz", "Pz"], sfreq=100, ch_types="eeg")
    epochs = mne.EpochsArray(eeg.reshape(1, 3, 500), info, verbose=False)
    pwr_ep = nk.eeg_power(epochs, frequency_band="Alpha")
    assert list(pwr_ep["Channel"]) == ["Fz", "Cz", "Pz"]

    evoked = epochs.average()
    df_ev = nk.mne_to_df(evoked)
    assert len(df_ev) == 500 and "Fz" in df_ev.columns

    with pytest.raises(ValueError, match="sampling_rate"):
        nk.eeg_power(eeg)

    # eeg_badchannels: legend label must name the bad channel (ch_names[bad], not ch_names[0]), and handle NaNs + Epochs
    bad_arr = rng.standard_normal((4, 200))
    bad_arr[2, :] *= 25.0
    bad_arr[0, 5] = np.nan
    df_bad = pd.DataFrame(bad_arr.T, columns=["Fp1", "Fp2", "Pz_BAD", "Oz"])
    bads, res_bad = nk.eeg_badchannels(df_bad, bad_threshold=0.3, distance_threshold=0.9, show=True)
    fig = plt.gcf()
    _, labels = fig.axes[0].get_legend_handles_labels()
    plt.close("all")
    assert bads == ["Pz_BAD"]
    assert labels == ["Pz_BAD"]
    assert np.all(np.isfinite(res_bad.loc[0, ["Skewness", "Kurtosis", "Amplitude"]]))

    bads_ep, _ = nk.eeg_badchannels(
        mne.EpochsArray(
            np.stack([bad_arr[:, :100], bad_arr[:, 100:]], axis=0),
            mne.create_info(["Fp1", "Fp2", "Pz_BAD", "Oz"], 100, "eeg"),
            verbose=False,
        ),
        bad_threshold=0.3,
        distance_threshold=0.9,
    )
    assert "Pz_BAD" in bads_ep

    # mne_crop must match MNE's Raw.crop() for both include_tmax=True and include_tmax=False
    raw = mne.io.RawArray(eeg, info, verbose=False)
    crop_incl = nk.mne_crop(raw, tmin=0.1, tmax=1.0, include_tmax=True)
    crop_excl = nk.mne_crop(raw, tmin=0.1, tmax=1.0, include_tmax=False)
    assert crop_incl.n_times == raw.copy().crop(0.1, 1.0, include_tmax=True).n_times
    assert crop_excl.n_times == raw.copy().crop(0.1, 1.0, include_tmax=False).n_times
    assert nk.mne_crop(raw, smin=10, smax=50, include_tmax=False).n_times == 40
    assert nk.mne_crop(raw, smin=10, smax=50, include_tmax=True).n_times == 41

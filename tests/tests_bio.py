import numpy as np
import pandas as pd
import pytest

import neurokit2 as nk


def test_bio_process():
    sampling_rate = 1000

    # Create data
    ecg = nk.ecg_simulate(duration=30, sampling_rate=sampling_rate)
    rsp = nk.rsp_simulate(duration=30, sampling_rate=sampling_rate)
    eda = nk.eda_simulate(duration=30, sampling_rate=sampling_rate, scr_number=3)
    emg = nk.emg_simulate(duration=30, sampling_rate=sampling_rate, burst_number=3)

    bio_df, bio_info = nk.bio_process(ecg=ecg, rsp=rsp, eda=eda, emg=emg, sampling_rate=sampling_rate)

    # SCR components
    scr = [val for key, val in bio_info.items() if "SCR" in key]
    assert all(len(elem) == len(scr[0]) for elem in scr)
    assert all(bio_info["SCR_Onsets"] < bio_info["SCR_Peaks"])
    assert all(bio_info["SCR_Peaks"] < bio_info["SCR_Recovery"])

    # RSP
    assert all(bio_info["RSP_Peaks"] > bio_info["RSP_Troughs"])
    assert len(bio_info["RSP_Peaks"]) == len(bio_info["RSP_Troughs"])

    # EMG
    assert all(bio_info["EMG_Offsets"] > bio_info["EMG_Onsets"])
    assert len(bio_info["EMG_Offsets"] == len(bio_info["EMG_Onsets"]))


def test_bio_analyze():
    # Example with event-related analysis
    data = nk.data("bio_eventrelated_100hz")
    df, info = nk.bio_process(
        ecg=data["ECG"],
        rsp=data["RSP"],
        eda=data["EDA"],
        keep=data["Photosensor"],
        sampling_rate=100,
    )
    events = nk.events_find(
        data["Photosensor"],
        threshold_keep="below",
        event_conditions=["Negative", "Neutral", "Neutral", "Negative"],
    )
    epochs = nk.epochs_create(df, events, sampling_rate=100, epochs_start=-0.1, epochs_end=1.9)
    event_related = nk.bio_analyze(epochs)

    assert len(event_related) == len(epochs)
    labels = [int(i) for i in event_related["Label"]]
    assert labels == list(np.arange(1, len(epochs) + 1))

    # Example with interval-related analysis
    data = nk.data("bio_resting_8min_100hz")
    df, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], eda=data["EDA"], sampling_rate=100)
    interval_related = nk.bio_analyze(df)

    assert len(interval_related) == 1


def test_analyze_epochs_dataframe_and_bio_analyze_edges():
    data = nk.data("bio_eventrelated_100hz")
    df, _ = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], eda=data["EDA"], sampling_rate=100)
    events = nk.events_find(data["Photosensor"], threshold_keep="below")

    # Custom string event labels + epochs_to_df DataFrame input across modality *_analyze and bio_analyze
    epochs = nk.epochs_create(
        df,
        events,
        sampling_rate=100,
        epochs_start=-0.1,
        epochs_end=1.9,
        event_labels=["trial_A", "trial_B", "trial_C", "trial_D"],
    )
    epochs_df = nk.epochs_to_df(epochs)

    assert len(nk.ecg_analyze(epochs_df, sampling_rate=100)) == 4
    assert len(nk.rsp_analyze(epochs_df, sampling_rate=100)) == 4
    assert len(nk.eda_analyze(epochs_df, sampling_rate=100)) == 4

    bio_ev_df = nk.bio_analyze(epochs_df, sampling_rate=100, window_lengths={"ECG": [-0.05, 1.5], "EDA": [0.0, 1.8]})
    assert len(bio_ev_df) == 4
    assert "RSA_P2T" in bio_ev_df.columns

    # Dict of long intervals in bio_analyze (exercises _bio_analyze_rsa_interval dict path without mutating caller)
    long_epochs = nk.epochs_create(df, events=[0, 5000], sampling_rate=100, epochs_start=0, epochs_end=45)
    orig_cols = list(long_epochs["1"].columns)
    bio_int_dict = nk.bio_analyze(long_epochs, sampling_rate=100, method="interval-related")
    assert len(bio_int_dict) == 2
    assert list(long_epochs["1"].columns) == orig_cols

    # PPG, EMG, EOG DataFrame epochs with Label column on pandas 3
    ppg_df, _ = nk.ppg_process(nk.ppg_simulate(duration=20, sampling_rate=100, random_state=42), sampling_rate=100)
    emg_df, _ = nk.emg_process(
        nk.emg_simulate(duration=10, sampling_rate=500, burst_number=2, random_state=42), sampling_rate=500
    )
    eog_df = pd.DataFrame({"EOG_Rate": np.full(2000, 15.0), "EOG_Blinks": np.zeros(2000, dtype=int)})
    eog_df.loc[[400, 1100], "EOG_Blinks"] = 1
    for mod_df, sr, fn in [(ppg_df, 100, nk.ppg_analyze), (emg_df, 500, nk.emg_analyze), (eog_df, 100, nk.eog_analyze)]:
        ep_mod = nk.epochs_to_df(
            nk.epochs_create(mod_df, events=[3 * sr, 6 * sr], sampling_rate=sr, epochs_start=-0.1, epochs_end=1.9)
        )
        assert len(fn(ep_mod, sampling_rate=sr)) == 2
        with pytest.raises(ValueError, match="not recognized"):
            fn(ep_mod, sampling_rate=sr, method="unknown")

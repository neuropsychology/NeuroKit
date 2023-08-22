import numpy as np

import neurokit2 as nk


def test_bio_process():
    sampling_rate = 1000

    # Create data
    ecg = nk.ecg_simulate(duration=30, sampling_rate=sampling_rate)
    rsp = nk.rsp_simulate(duration=30, sampling_rate=sampling_rate)
    eda = nk.eda_simulate(duration=30, sampling_rate=sampling_rate, scr_number=3)
    emg = nk.emg_simulate(duration=30, sampling_rate=sampling_rate, burst_number=3)

    bio_df, bio_info = nk.bio_process(
        ecg=ecg, rsp=rsp, eda=eda, emg=emg, sampling_rate=sampling_rate
    )

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
    epochs = nk.epochs_create(
        df, events, sampling_rate=100, epochs_start=-0.1, epochs_end=1.9
    )
    event_related = nk.bio_analyze(epochs)

    assert len(event_related) == len(epochs)
    labels = [int(i) for i in event_related["Label"]]
    assert labels == list(np.arange(1, len(epochs) + 1))

    # Example with interval-related analysis
    data = nk.data("bio_resting_8min_100hz")
    df, info = nk.bio_process(
        ecg=data["ECG"], rsp=data["RSP"], eda=data["EDA"], sampling_rate=100
    )
    interval_related = nk.bio_analyze(df)

    assert len(interval_related) == 1

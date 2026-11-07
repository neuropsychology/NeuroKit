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


@pytest.fixture(scope="module")
def cardio_signals():
    # Different beat intervals make accidental substitution of ECG for PPG visible.
    rng = np.random.default_rng(42)
    signals = pd.DataFrame({"ECG_Rate": np.full(12000, 70.0), "PPG_Rate": np.full(12000, 80.0)})
    for column, low, high in [("ECG_R_Peaks", 75, 95), ("PPG_Peaks", 65, 85)]:
        peaks = np.cumsum(rng.integers(low, high, size=200))
        signals[column] = 0
        signals.loc[peaks[peaks < len(signals)], column] = 1
    return signals


@pytest.mark.parametrize("method", ["auto", "interval-related"])
@pytest.mark.parametrize("as_epochs", [False, True])
def test_bio_analyze_ecg_ppg_hrv(cardio_signals, method, as_epochs):
    data = cardio_signals
    if as_epochs:
        data = nk.epochs_create(data, events=[0, 6000], sampling_rate=100, epochs_end=60)

    combined = nk.bio_analyze(data, sampling_rate=100, method=method)
    ecg = nk.ecg_analyze(data, sampling_rate=100, method=method)
    ppg = nk.ppg_analyze(data, sampling_rate=100, method=method)
    ppg = ppg.rename(columns={col: "PPG_" + col for col in ppg if col.startswith("HRV_")})

    assert combined.columns.is_unique
    pd.testing.assert_frame_equal(combined[ecg.columns], ecg)
    pd.testing.assert_frame_equal(combined[ppg.columns], ppg)
    assert isinstance(combined["HRV_MeanNN"], pd.Series)
    assert isinstance(combined["PPG_HRV_MeanNN"], pd.Series)
    assert not np.allclose(combined["HRV_MeanNN"], combined["PPG_HRV_MeanNN"])


@pytest.mark.parametrize("signal", ["ECG", "PPG"])
@pytest.mark.parametrize("as_epochs", [False, True])
def test_bio_analyze_single_cardio_signal(cardio_signals, signal, as_epochs):
    data = cardio_signals.filter(like=signal)
    if as_epochs:
        data = nk.epochs_create(data, events=[0], sampling_rate=100, epochs_end=120)

    analyze = nk.ecg_analyze if signal == "ECG" else nk.ppg_analyze
    expected = analyze(data, sampling_rate=100)
    actual = nk.bio_analyze(data, sampling_rate=100)

    pd.testing.assert_frame_equal(actual, expected)
    assert "HRV_MeanNN" in actual
    assert "PPG_HRV_MeanNN" not in actual


def test_bio_analyze_ecg_ppg_eventrelated(cardio_signals):
    epochs = nk.epochs_create(
        cardio_signals,
        events=[2000, 6000],
        event_conditions=["A", "B"],
        sampling_rate=100,
        epochs_start=-0.1,
        epochs_end=1.9,
    )
    actual = nk.bio_analyze(epochs, sampling_rate=100, method="event-related")
    ecg = nk.ecg_analyze(epochs, sampling_rate=100, method="event-related")
    ppg = nk.ppg_analyze(epochs, sampling_rate=100, method="event-related")

    assert actual.columns.is_unique
    pd.testing.assert_frame_equal(actual[ecg.columns], ecg)
    pd.testing.assert_frame_equal(actual[ppg.columns], ppg)
    assert not any("HRV_" in col for col in actual)

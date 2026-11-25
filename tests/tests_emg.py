import biosppy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.signal
import scipy.stats

import neurokit2 as nk


# =============================================================================
# EMG
# =============================================================================


def test_emg_simulate():
    emg1 = nk.emg_simulate(duration=20, length=5000, burst_number=1)
    assert len(emg1) == 5000

    emg2 = nk.emg_simulate(duration=20, length=5000, burst_number=15)
    assert scipy.stats.median_abs_deviation(emg1) < scipy.stats.median_abs_deviation(emg2)

    emg3 = nk.emg_simulate(duration=20, length=5000, burst_number=1, burst_duration=2.0)
    #    pd.DataFrame({"EMG1":emg1, "EMG3": emg3}).plot()
    assert len(nk.signal_findpeaks(emg3, height_min=1.0)["Peaks"]) > len(nk.signal_findpeaks(emg1, height_min=1.0)["Peaks"])


def test_emg_activation():
    emg = nk.emg_simulate(duration=10, burst_number=3)
    cleaned = nk.emg_clean(emg)
    emg_amplitude = nk.emg_amplitude(cleaned)

    activity_signal, info = nk.emg_activation(emg_amplitude)

    assert set(activity_signal.columns.to_list()) == set(list(info.keys()))
    assert len(info["EMG_Onsets"]) == len(info["EMG_Offsets"])
    for i, j in zip(info["EMG_Onsets"], info["EMG_Offsets"]):
        assert i < j


@pytest.mark.parametrize("burst_number", [1, 3])
def test_emg_activation_pelt_labels_bursts(burst_number):
    emg = nk.emg_simulate(duration=10, sampling_rate=500, burst_number=burst_number, random_state=0)
    cleaned = nk.emg_clean(emg, sampling_rate=500)
    signals, _ = nk.emg_activation(emg_cleaned=cleaned, sampling_rate=500, method="pelt")
    active = signals["EMG_Activity"].to_numpy() == 1
    assert cleaned[active].std() > cleaned[~active].std()


def test_emg_clean():
    sampling_rate = 1000

    emg = nk.emg_simulate(duration=20, sampling_rate=sampling_rate)
    emg_cleaned = nk.emg_clean(emg, sampling_rate=sampling_rate)

    assert emg.size == emg_cleaned.size

    # Comparison to biosppy (https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/emg.py)
    original, _, _ = biosppy.tools.filter_signal(
        signal=emg,
        ftype="butter",
        band="highpass",
        order=4,
        frequency=100,
        sampling_rate=sampling_rate,
    )
    emg_cleaned_biosppy = nk.signal_detrend(original, order=0)
    assert np.allclose((emg_cleaned - emg_cleaned_biosppy).mean(), 0, atol=1e-6)


def test_emg_plot():
    sampling_rate = 1000

    emg = nk.emg_simulate(duration=10, sampling_rate=1000, burst_number=3)
    emg_summary, info = nk.emg_process(emg, sampling_rate=sampling_rate)

    # Plot data over samples.
    nk.emg_plot(emg_summary, info)
    fig = plt.gcf()
    assert len(fig.axes) == 2
    titles = ["Raw and Cleaned Signal", "Muscle Activation"]
    for ax, title in zip(fig.get_axes(), titles):
        assert ax.get_title() == title
    assert fig.get_axes()[1].get_xlabel() == "Time (seconds)"
    assert np.array_equal(fig.axes[0].get_xticks(), fig.axes[1].get_xticks())
    plt.close(fig)


def test_emg_eventrelated():
    emg = nk.emg_simulate(duration=20, sampling_rate=1000, burst_number=3)
    emg_signals, info = nk.emg_process(emg, sampling_rate=1000)
    epochs = nk.epochs_create(
        emg_signals,
        events=[3000, 6000, 9000],
        sampling_rate=1000,
        epochs_start=-0.1,
        epochs_end=1.9,
    )
    emg_eventrelated = nk.emg_eventrelated(epochs)

    # Test amplitude features
    no_activation = np.where(emg_eventrelated["EMG_Activation"] == 0)[0][0]
    assert int(pd.DataFrame(emg_eventrelated.values[no_activation]).isna().sum().iloc[0]) == 5

    assert np.all(
        np.nansum(np.array(emg_eventrelated["EMG_Amplitude_Mean"]))
        < np.nansum(np.array(emg_eventrelated["EMG_Amplitude_Max"]))
    )

    assert len(emg_eventrelated["Label"]) == 3

    # Test warning on missing columns
    with pytest.warns(nk.misc.NeuroKitWarning, match=r".*does not have an `EMG_Onsets`.*"):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["EMG_Onsets"]
        nk.emg_eventrelated({**epochs, first_epoch_key: first_epoch_copy})

    with pytest.warns(nk.misc.NeuroKitWarning, match=r".*does not have an `EMG_Activity`.*"):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["EMG_Activity"]
        nk.emg_eventrelated({**epochs, first_epoch_key: first_epoch_copy})

    with pytest.warns(nk.misc.NeuroKitWarning, match=r".*does not have an.*`EMG_Amplitude`.*"):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["EMG_Amplitude"]
        nk.emg_eventrelated({**epochs, first_epoch_key: first_epoch_copy})


def test_emg_intervalrelated():
    emg = nk.emg_simulate(duration=40, sampling_rate=1000, burst_number=3)
    emg_signals, info = nk.emg_process(emg, sampling_rate=1000)
    columns = ["EMG_Activation_N", "EMG_Amplitude_Mean"]

    # Test with signal dataframe
    features_df = nk.emg_intervalrelated(emg_signals)

    assert all(elem in columns for elem in np.array(features_df.columns.values, dtype=str))
    assert features_df.shape[0] == 1  # Number of rows

    # Test with dict
    columns.append("Label")
    epochs = nk.epochs_create(emg_signals, events=[0, 20000], sampling_rate=1000, epochs_end=20)
    features_dict = nk.emg_intervalrelated(epochs)

    assert all(elem in columns for elem in np.array(features_dict.columns.values, dtype=str))
    assert features_dict.shape[0] == 2  # Number of rows


def test_emg_intervalrelated_formatinput_fresh_output():
    from neurokit2.emg.emg_intervalrelated import _emg_intervalrelated_formatinput

    emg = nk.emg_simulate(duration=10, sampling_rate=1000, burst_number=2, random_state=1)
    emg_signals, _ = nk.emg_process(emg, sampling_rate=1000)

    first = _emg_intervalrelated_formatinput(emg_signals)
    first["Extra"] = 1
    second = _emg_intervalrelated_formatinput(emg_signals)

    # Each call must return its own dict, not a shared default
    assert second is not first
    assert "Extra" not in second


@pytest.mark.parametrize(
    "method_cleaning, method_activation, threshold",
    [
        ("none", "threshold", "default"),
        ("biosppy", "pelt", 0.5),
        ("biosppy", "mixture", 0.05),
        ("biosppy", "biosppy", "default"),
        ("biosppy", "silva", "default"),
    ],
)
def test_emg_report(tmp_path, method_cleaning, method_activation, threshold):
    sampling_rate = 250

    emg = nk.emg_simulate(
        duration=30,
        sampling_rate=sampling_rate,
        random_state=0,
    )

    d = tmp_path / "sub"
    d.mkdir()
    p = d / "myreport.html"

    signals, _ = nk.emg_process(
        emg,
        sampling_rate=sampling_rate,
        report=str(p),
        method_cleaning=method_cleaning,
        method_activation=method_activation,
        threshold=threshold,
    )
    assert p.is_file()
    assert "EMG_Activity" in signals.columns


def test_emg_process_amplitude_sampling_rate():
    sampling_rate = 4000
    emg = nk.emg_simulate(duration=20, sampling_rate=sampling_rate, burst_number=4, random_state=42)
    signals, _ = nk.emg_process(emg, sampling_rate=sampling_rate)
    wrong = nk.emg_amplitude(signals["EMG_Clean"].to_numpy(), sampling_rate=1000)

    amplitudes = np.array([signals["EMG_Amplitude"], wrong])
    freqs, psd = scipy.signal.welch(amplitudes, fs=sampling_rate, nperseg=4 * sampling_rate)
    correct_ratio, wrong_ratio = psd[:, freqs > 20].sum(axis=1) / psd.sum(axis=1)

    assert correct_ratio < 1e-4 < wrong_ratio


def test_emg_process_1000_hz_amplitude_is_unchanged():
    emg = nk.emg_simulate(duration=10, sampling_rate=1000, random_state=42)
    signals, _ = nk.emg_process(emg, sampling_rate=1000)

    assert np.allclose(signals["EMG_Amplitude"], nk.emg_amplitude(signals["EMG_Clean"].to_numpy()))


def test_emg_process_below_amplitude_highcut_nyquist():
    sampling_rate = 250
    emg = nk.emg_simulate(duration=2, sampling_rate=sampling_rate, random_state=42)

    signals, _ = nk.emg_process(emg, sampling_rate=sampling_rate)

    assert len(signals) == len(emg)
    assert np.isfinite(signals["EMG_Amplitude"]).all()


def test_emg_intervalrelated_missing_amplitude():
    data = pd.DataFrame({"EMG_Onsets": [0, 1, 0, 0], "EMG_Activity": [0, 1, 1, 0], "Label": ["1"] * 4})
    with pytest.raises(ValueError, match="EMG_Amplitude"):
        nk.emg_intervalrelated(data)
    with pytest.raises(ValueError, match="EMG_Amplitude"):
        nk.emg_intervalrelated({"1": data})


def test_emg_activation_trailing_and_silva_edge_cases():
    # Invalid method error includes 'silva'
    with pytest.raises(ValueError, match="silva"):
        nk.emg_activation(emg_amplitude=np.ones(100), method="invalid")

    # Trailing burst extending to the end of the signal keeps paired onset/offset and active tail
    amp = np.zeros(200)
    amp[120:] = 1.0
    signals, info = nk.emg_activation(emg_amplitude=amp, sampling_rate=1000, method="threshold", threshold=0.5)
    assert len(info["EMG_Onsets"]) == len(info["EMG_Offsets"]) == 1
    assert info["EMG_Onsets"][0] == 120
    assert info["EMG_Offsets"][0] == 199
    assert signals["EMG_Activity"].iloc[-1] == 1
    assert signals["EMG_Offsets"].iloc[-1] == 1

    # Silva method on quiet signal (0 bursts), short signal (< threshold_size), and unclosed trailing burst
    quiet_sig, quiet_info = nk.emg_activation(emg_cleaned=np.zeros(200), sampling_rate=1000, method="silva")
    assert quiet_info["EMG_Onsets"].size == 0
    assert quiet_info["EMG_Offsets"].size == 0
    assert (quiet_signals := quiet_sig["EMG_Activity"]).sum() == 0
    assert quiet_signals.dtype.kind in ("i", "u")

    short_sig, short_info = nk.emg_activation(emg_cleaned=np.zeros(10), sampling_rate=1000, method="silva")
    assert short_info["EMG_Activity"].size == 0
    assert len(short_sig) == 10

    rng = np.random.default_rng(0)
    trailing = np.zeros(500)
    trailing[300:] = rng.normal(0, 1.0, size=200)
    silva_sig, silva_info = nk.emg_activation(emg_cleaned=trailing, sampling_rate=1000, method="silva", threshold=0.05)
    assert len(silva_info["EMG_Onsets"]) == len(silva_info["EMG_Offsets"]) >= 1
    assert silva_sig["EMG_Activity"].sum() > 50


def test_emg_clean_missing_and_low_sampling_rate():
    rng = np.random.default_rng(42)
    raw = rng.normal(0, 1, size=200) + 5.0
    raw_nan = raw.copy()
    raw_nan[0] = np.nan
    raw_nan[25] = np.nan

    with pytest.warns(nk.misc.NeuroKitWarning, match="missing data"):
        cleaned_nan = nk.emg_clean(raw_nan, sampling_rate=1000)
    assert np.all(np.isfinite(cleaned_nan))

    # Sampling rate <= 200 Hz warns and detrends instead of crashing scipy.signal.butter
    with pytest.warns(nk.misc.NeuroKitWarning, match="too low"):
        cleaned_low = nk.emg_clean(raw, sampling_rate=100)
    assert cleaned_low.shape == raw.shape
    assert np.isclose(np.mean(cleaned_low), 0.0, atol=1e-10)


def test_emg_eventrelated_intervalrelated_and_plot_edge_cases():
    from neurokit2.emg.emg_eventrelated import _emg_eventrelated_features

    # NaN padding in post-event EMG_Onsets must not trigger EMG_Activation == 1
    idx = np.linspace(-0.1, 0.5, 7)
    ep_nan = pd.DataFrame(
        {
            "EMG_Onsets": [0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan],
            "EMG_Activity": [0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan],
            "EMG_Amplitude": [0.1, 0.1, 0.1, 0.1, 0.1, np.nan, np.nan],
            "Label": ["1"] * 7,
        },
        index=idx,
    )
    res_nan = nk.emg_eventrelated({"1": ep_nan})
    assert res_nan["EMG_Activation"].iloc[0] == 0
    assert np.isnan(res_nan["EMG_Amplitude_Mean"].iloc[0])

    # Active burst max time comes from active region even if inactive sample matches max amplitude
    ep_active = pd.DataFrame(
        {
            "EMG_Onsets": [0, 0, 0, 1, 0, 0, 0],
            "EMG_Activity": [0, 0, 0, 1, 1, 0, 0],
            "EMG_Amplitude": [0.1, 0.1, 5.0, 2.0, 5.0, 0.1, 0.1],
            "Label": ["1"] * 7,
        },
        index=idx,
    )
    res_active = nk.emg_eventrelated({"1": ep_active})
    assert res_active["EMG_Activation"].iloc[0] == 1
    assert np.isclose(res_active["EMG_Amplitude_Max_Time"].iloc[0], idx[4])

    # Fresh default dict and zero-active / all-NaN active guards in _emg_eventrelated_features
    ep_no_act = ep_active.copy()
    ep_no_act["EMG_Activity"] = 0
    out1 = _emg_eventrelated_features(ep_no_act)
    out1["Extra"] = 1
    out2 = _emg_eventrelated_features(ep_no_act)
    assert "Extra" not in out2
    assert np.isnan(out2["EMG_Amplitude_Mean"])

    # emg_intervalrelated with 0 bursts (no RuntimeWarning) and dict without Label column
    zero_df = pd.DataFrame(
        {
            "EMG_Raw": np.zeros(50),
            "EMG_Clean": np.zeros(50),
            "EMG_Amplitude": np.ones(50) * 0.2,
            "EMG_Activity": np.zeros(50, dtype=int),
            "EMG_Onsets": np.zeros(50, dtype=int),
            "EMG_Offsets": np.zeros(50, dtype=int),
        }
    )
    iv_df = nk.emg_intervalrelated(zero_df)
    assert iv_df["EMG_Activation_N"].iloc[0] == 0
    assert np.isnan(iv_df["EMG_Amplitude_Mean"].iloc[0])

    iv_dict = nk.emg_intervalrelated({"a": zero_df})
    assert iv_dict.loc["a", "EMG_Activation_N"] == 0
    assert np.isnan(iv_dict.loc["a", "EMG_Amplitude_Mean"])

    with pytest.raises(ValueError, match="DataFrame or a dict"):
        nk.emg_intervalrelated([1, 2, 3])

    # emg_plot with 0 bursts, sliced DataFrame (non-zero index), and invalid input
    with pytest.raises(ValueError, match="DataFrame"):
        nk.emg_plot(np.zeros(10), {"sampling_rate": 100})

    nk.emg_plot(zero_df, {"sampling_rate": 100})
    plt.close("all")

    emg = nk.emg_simulate(duration=5, sampling_rate=500, burst_number=2, random_state=42)
    emg_signals, info = nk.emg_process(emg, sampling_rate=500)
    nk.emg_plot(emg_signals.iloc[100:2400], info)
    plt.close("all")

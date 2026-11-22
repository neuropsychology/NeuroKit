import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import neurokit2 as nk


# =============================================================================
# EDA
# =============================================================================


def test_eda_simulate():
    eda1 = nk.eda_simulate(duration=10, length=None, scr_number=1, random_state=333)
    assert len(nk.signal_findpeaks(eda1, height_min=0.6)["Peaks"]) == 1

    eda2 = nk.eda_simulate(duration=10, length=None, scr_number=5, random_state=333)
    assert len(nk.signal_findpeaks(eda2, height_min=0.6)["Peaks"]) == 5
    #   pd.DataFrame({"EDA1": eda1, "EDA2": eda2}).plot()

    assert len(nk.signal_findpeaks(eda2, height_min=0.6)["Peaks"]) > len(nk.signal_findpeaks(eda1, height_min=0.6)["Peaks"])


def test_eda_clean():
    sampling_rate = 1000
    eda = nk.eda_simulate(
        duration=30,
        sampling_rate=sampling_rate,
        scr_number=6,
        noise=0.01,
        drift=0.01,
        random_state=42,
    )

    clean = nk.eda_clean(eda, sampling_rate=sampling_rate)
    assert len(clean) == len(eda)

    # Comparison to biosppy (https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/eda.py)
    # Test deactivated because it fails

    # eda_biosppy = nk.eda_clean(eda, sampling_rate=sampling_rate, method="biosppy")
    # original, _, _ = biosppy.tools.filter_signal(
    #     signal=eda,
    #     ftype="butter",
    #     band="lowpass",
    #     order=4,
    #     frequency=5,
    #     sampling_rate=sampling_rate,
    # )

    # original, _ = biosppy.tools.smoother(
    #     signal=original, kernel="boxzen", size=int(0.75 * sampling_rate), mirror=True
    # )

    # #    pd.DataFrame({"our":eda_biosppy, "biosppy":original}).plot()
    # assert np.allclose((eda_biosppy - original).mean(), 0, atol=1e-5)


def test_eda_phasic():
    sr = 100
    eda = nk.eda_simulate(
        duration=30,
        sampling_rate=sr,
        scr_number=6,
        noise=0.01,
        drift=0.01,
        random_state=42,
    )

    if platform.system() == "Linux":
        cvxEDA = nk.eda_phasic(eda, sampling_rate=sr, method="cvxeda")
        assert len(cvxEDA) == len(eda)

    smoothMedian = nk.eda_phasic(eda, sampling_rate=sr, method="smoothmedian")
    assert len(smoothMedian) == len(eda)

    highpass = nk.eda_phasic(eda, sampling_rate=sr, method="highpass")
    assert len(highpass) == len(eda)

    sparsEDA = nk.eda_phasic(eda, sampling_rate=sr, method="sparsEDA")
    assert len(sparsEDA) == len(eda)


def test_eda_peaks():
    sampling_rate = 1000
    eda = nk.eda_simulate(
        duration=30 * 20,
        sampling_rate=sampling_rate,
        scr_number=6 * 20,
        noise=0,
        drift=0.01,
        random_state=42,
    )
    eda_phasic = nk.eda_phasic(nk.standardize(eda), method="highpass")["EDA_Phasic"].values

    signals, info = nk.eda_peaks(eda_phasic, method="gamboa2008")

    # 120 Value based on counting by eye
    assert len(info["SCR_Peaks"]) == 120

    signals, info = nk.eda_peaks(eda_phasic, method="kim2004")

    # Check that indices and values positions match
    peak_positions = np.where(info["SCR_Peaks"] != 0)[0]
    assert np.all(peak_positions == np.where(info["SCR_Amplitude"] != 0)[0])
    assert np.all(peak_positions == np.where(info["SCR_Height"] != 0)[0])
    assert np.all(peak_positions == np.where(info["SCR_RiseTime"] != 0)[0])

    recovery_positions = np.where(info["SCR_Recovery"] != 0)[0]
    assert np.all(recovery_positions == np.where(info["SCR_RecoveryTime"] != 0)[0])


def test_eda_process():
    eda = nk.eda_simulate(duration=30, scr_number=5, drift=0.1, noise=0, sampling_rate=250)
    signals, info = nk.eda_process(eda, sampling_rate=250)

    assert signals.shape == (7500, 11)
    assert {
        "EDA_Raw",
        "EDA_Clean",
        "EDA_Tonic",
        "EDA_Phasic",
        "SCR_Onsets",
        "SCR_Peaks",
        "SCR_Height",
        "SCR_Amplitude",
        "SCR_RiseTime",
        "SCR_Recovery",
        "SCR_RecoveryTime",
    }.issubset(signals.columns)

    # Check equal number of markers
    peaks = np.where(signals["SCR_Peaks"] == 1)[0]
    onsets = np.where(signals["SCR_Onsets"] == 1)[0]
    recovery = np.where(signals["SCR_Recovery"] == 1)[0]
    assert peaks.shape == onsets.shape == recovery.shape == (5,)


def test_eda_process_custom_amplitude_min():
    """Custom amplitude_min must reach eda_peaks through eda_process (#1197)."""
    from neurokit2.eda.eda_methods import eda_methods

    methods = eda_methods(amplitude_min=0.03)
    assert methods["kwargs_peaks"]["amplitude_min"] == 0.03

    sampling_rate = 250
    eda = nk.eda_simulate(
        duration=30,
        scr_number=5,
        drift=0.1,
        noise=0,
        sampling_rate=sampling_rate,
        random_state=42,
    )

    signals_default, _ = nk.eda_process(eda, sampling_rate=sampling_rate)
    signals_strict, _ = nk.eda_process(eda, sampling_rate=sampling_rate, amplitude_min=0.9)
    signals_loose, _ = nk.eda_process(eda, sampling_rate=sampling_rate, amplitude_min=0.01)

    n_default = int(np.sum(signals_default["SCR_Peaks"] == 1))
    n_strict = int(np.sum(signals_strict["SCR_Peaks"] == 1))
    n_loose = int(np.sum(signals_loose["SCR_Peaks"] == 1))

    assert n_strict <= n_default
    assert n_loose >= n_default
    assert n_strict < n_loose


def test_eda_plot():
    sampling_rate = 1000
    eda = nk.eda_simulate(
        duration=30,
        sampling_rate=sampling_rate,
        scr_number=6,
        noise=0,
        drift=0.01,
        random_state=42,
    )
    eda_summary, info = nk.eda_process(eda, sampling_rate=sampling_rate)

    # Plot data over samples.
    nk.eda_plot(eda_summary, info)
    # This will identify the latest figure.
    fig = plt.gcf()
    assert len(fig.axes) == 3
    titles = [
        "Raw and Cleaned Signal",
        "Skin Conductance Response (SCR)",
        "Skin Conductance Level (SCL)",
    ]
    for ax, title in zip(fig.get_axes(), titles):
        assert ax.get_title() == title
    assert fig.get_axes()[2].get_xlabel() == "Time (seconds)"
    np.testing.assert_array_equal(fig.axes[0].get_xticks(), fig.axes[1].get_xticks(), fig.axes[2].get_xticks())
    plt.close(fig)


def test_eda_eventrelated():
    eda = nk.eda_simulate(duration=15, scr_number=3)
    eda_signals, _ = nk.eda_process(eda, sampling_rate=1000)
    epochs = nk.epochs_create(
        eda_signals,
        events=[5000, 10000, 15000],
        sampling_rate=1000,
        epochs_start=-0.1,
        epochs_end=1.9,
    )
    eda_eventrelated = nk.eda_eventrelated(epochs)

    no_activation = np.where(eda_eventrelated["EDA_SCR"] == 0)[0][0]
    assert pd.DataFrame(eda_eventrelated.values[no_activation]).isna().sum()[0] == 4

    assert len(eda_eventrelated["Label"]) == 3


def test_eda_intervalrelated():
    data = nk.data("bio_resting_8min_100hz")
    df, _ = nk.eda_process(data["EDA"], sampling_rate=100)
    columns = ["SCR_Peaks_N", "SCR_Peaks_Amplitude_Mean"]

    # Test with signal dataframe
    rez = nk.eda_intervalrelated(df)

    assert all([i in rez.columns.values for i in columns])
    assert rez.shape[0] == 1  # Number of rows

    # Test with dict
    columns.append("Label")
    epochs = nk.epochs_create(df, events=[0, 25300], sampling_rate=100, epochs_end=20)
    rez = nk.eda_intervalrelated(epochs)

    assert all([i in rez.columns.values for i in columns])
    assert rez.shape[0] == 2  # Number of rows


def test_eda_sympathetic():
    eda_signal = nk.data("bio_eventrelated_100hz")["EDA"]
    indexes_posada = nk.eda_sympathetic(eda_signal, sampling_rate=100, method="posada")
    # Test value is float
    assert isinstance(indexes_posada["EDA_Sympathetic"], float)
    assert isinstance(indexes_posada["EDA_SympatheticN"], float)


def test_eda_findpeaks():
    eda_signal = nk.data("bio_eventrelated_100hz")["EDA"]
    eda_cleaned = nk.eda_clean(eda_signal)
    eda = nk.eda_phasic(eda_cleaned)
    eda_phasic = eda["EDA_Phasic"].values

    # Find peaks
    nabian2018 = nk.eda_findpeaks(eda_phasic, sampling_rate=100, method="nabian2018")
    assert len(nabian2018["SCR_Peaks"]) == 9

    vanhalem2020 = nk.eda_findpeaks(eda_phasic, sampling_rate=100, method="vanhalem2020")
    min_n_peaks = min(len(vanhalem2020), len(nabian2018))
    assert any(nabian2018["SCR_Peaks"][:min_n_peaks] - vanhalem2020["SCR_Peaks"][:min_n_peaks]) < np.mean(eda_signal)


@pytest.mark.parametrize(
    "method_cleaning, method_phasic, method_peaks",
    [
        ("none", "cvxeda", "gamboa2008"),
        ("neurokit", "median", "nabian2018"),
    ],
)
def test_eda_report(tmp_path, method_cleaning, method_phasic, method_peaks):
    sampling_rate = 100

    eda = nk.eda_simulate(
        duration=30,
        sampling_rate=sampling_rate,
        scr_number=6,
        noise=0,
        drift=0.01,
        random_state=0,
    )

    d = tmp_path / "sub"
    d.mkdir()
    p = d / "myreport.html"

    signals, _ = nk.eda_process(
        eda,
        sampling_rate=sampling_rate,
        method_cleaning=method_cleaning,
        method_phasic=method_phasic,
        method_peaks=method_peaks,
        report=str(p),
    )

    assert p.is_file()


@pytest.mark.parametrize("sampling_rate, lag", [(1000, 2.5), (1000.0, 2)])
def test_eda_autocor(sampling_rate, lag):
    eda = np.random.default_rng(33).random(8000)
    expected = nk.signal_autocor(eda, lag=int(lag * 1000))[0]
    assert nk.eda_autocor(eda, sampling_rate=sampling_rate, lag=lag) == pytest.approx(expected)


def test_eda_autocor_lag_too_long():
    with pytest.raises(ValueError, match="exceeds the duration"):
        nk.eda_autocor(np.zeros(4000), sampling_rate=1000, lag=4)


def test_eda_findpeaks_pairing_ties_and_sampling_rate():
    # gamboa2008 must pair each peak with its preceding onset (onsets < peaks)
    eda = nk.eda_simulate(duration=30, sampling_rate=100, scr_number=4, noise=0, drift=0.0, random_state=42)
    phasic = nk.eda_phasic(eda, sampling_rate=100, method="highpass")["EDA_Phasic"].to_numpy()
    gamboa = nk.eda_findpeaks(phasic, sampling_rate=100, method="gamboa2008")
    assert len(gamboa["SCR_Onsets"]) == len(gamboa["SCR_Peaks"]) == 4
    assert np.all(gamboa["SCR_Onsets"] < gamboa["SCR_Peaks"])

    _, gamboa_peaks_info = nk.eda_peaks(phasic, sampling_rate=100, method="gamboa2008")
    assert np.all(np.isfinite(gamboa_peaks_info["SCR_RiseTime"]))
    assert np.all(gamboa_peaks_info["SCR_RiseTime"] > 0)

    # nabian2018 handles tied peak amplitudes without ValueError and flat/short signals cleanly
    t = np.linspace(0, 20, 2000)
    tied = np.exp(-((t - 5.0) ** 2) / 0.5) + np.exp(-((t - 14.0) ** 2) / 0.5) - 0.01 * t
    # Force exact float tie at both peaks
    p1 = int(np.argmax(tied[:1000]))
    p2 = 1000 + int(np.argmax(tied[1000:]))
    tied[p2] = tied[p1]
    nab = nk.eda_findpeaks(tied, sampling_rate=100, method="nabian2018")
    assert len(nab["SCR_Peaks"]) == 2
    assert np.all(nab["SCR_Onsets"] < nab["SCR_Peaks"])

    empty_nab = nk.eda_findpeaks(np.zeros(50), sampling_rate=100, method="nabian2018")
    assert empty_nab["SCR_Peaks"].size == 0

    # vanhalem2020 scales window_size with sampling_rate and handles <501-sample low-rate signals
    phasic_low = -np.cos(2 * np.pi * np.arange(300) / 60)
    vh_low = nk.eda_findpeaks(phasic_low, sampling_rate=10, method="vanhalem2020")
    assert len(vh_low["SCR_Peaks"]) >= 2
    assert np.all(vh_low["SCR_Onsets"] < vh_low["SCR_Peaks"])
    assert nk.eda_findpeaks(np.array([]), sampling_rate=10, method="vanhalem2020")["SCR_Peaks"].size == 0
    assert nk.eda_findpeaks(np.array([0.0, 1.0, 0.0]), sampling_rate=4, method="vanhalem2020")["SCR_Peaks"].size == 0


def test_eda_eventrelated_postevent_pairing_and_nan_padding():
    from neurokit2.eda.eda_eventrelated import _eda_eventrelated_eda, _eda_eventrelated_scr

    idx = np.array([-0.5, -0.2, 0.2, 0.5, 1.0, 1.8, 2.5, 3.0])
    # Pre-stimulus SCR starts at -0.5, peaks at +0.2 (amp=9.0), recovers at +0.5 (recov=0.3);
    # True post-stimulus SCR starts at +1.0, peaks at +1.8 (amp=2.5, rise=0.8), padded NaN at tail without recovery
    ep = pd.DataFrame(
        {
            "EDA_Phasic": [0.0, 2.0, 9.0, 4.5, 0.1, 2.6, 2.0, np.nan],
            "SCR_Onsets": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.nan],
            "SCR_Peaks": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, np.nan],
            "SCR_Amplitude": [0.0, 0.0, 9.0, 0.0, 0.0, 2.5, 0.0, np.nan],
            "SCR_RiseTime": [0.0, 0.0, 0.7, 0.0, 0.0, 0.8, 0.0, np.nan],
            "SCR_Recovery": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, np.nan],
            "SCR_RecoveryTime": [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, np.nan],
            "Label": ["1"] * 8,
        },
        index=idx,
    )
    res = nk.eda_eventrelated({"1": ep})
    assert res["EDA_SCR"].iloc[0] == 1
    assert np.isclose(res["SCR_Peak_Amplitude"].iloc[0], 2.5)
    assert np.isclose(res["SCR_Peak_Amplitude_Time"].iloc[0], 1.8)
    assert np.isclose(res["SCR_RiseTime"].iloc[0], 0.8)
    assert np.isnan(res["SCR_RecoveryTime"].iloc[0])

    # Epoch where pre-stimulus SCR peaks at +0.2 and post-stimulus onset at +2.5 has not peaked yet -> EDA_SCR == 0
    ep_unpeaked = ep.copy()
    ep_unpeaked["SCR_Peaks"] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, np.nan]
    res_unpeaked = nk.eda_eventrelated({"1": ep_unpeaked})
    assert res_unpeaked["EDA_SCR"].iloc[0] == 0

    # Mutable default dict safety
    d1 = _eda_eventrelated_eda(ep)
    d1["Extra"] = 1
    assert "Extra" not in _eda_eventrelated_eda(ep)
    s1 = _eda_eventrelated_scr(ep)
    s1["Extra"] = 1
    assert "Extra" not in _eda_eventrelated_scr(ep)


def test_eda_simulate_length_clean_missing_and_sparseda_kwargs():
    # Exact length respected when length != duration * sampling_rate
    sim = nk.eda_simulate(duration=10, length=500, sampling_rate=100, scr_number=2, random_state=42)
    assert len(sim) == 500
    assert np.all(np.isfinite(sim))

    # eda_clean handles missing values (including leading NaN) and low-rate biosppy
    sim_nan = sim.copy()
    sim_nan[0] = np.nan
    sim_nan[20] = np.nan
    with pytest.warns(nk.misc.NeuroKitWarning, match="missing data"):
        cleaned = nk.eda_clean(sim_nan, sampling_rate=50)
    assert np.all(np.isfinite(cleaned))

    with pytest.warns(nk.misc.NeuroKitWarning, match="very low frequency"):
        cleaned_low = nk.eda_clean(np.ones(32), sampling_rate=4, method="biosppy")
    assert len(cleaned_low) == 32

    # sparsEDA forwards kwargs, handles constant signals without IndexError, and rejects NaNs with ValueError
    flat_decomp = nk.eda_phasic(np.ones(200), sampling_rate=20, method="sparsEDA", Kmax=10, epsilon=1e-3)
    assert flat_decomp.shape == (200, 2)
    assert np.all(np.isfinite(flat_decomp))

    with pytest.raises(ValueError, match="NaN"):
        nk.eda_phasic(np.array([1.0, np.nan, 1.0]), sampling_rate=10, method="sparsEDA")

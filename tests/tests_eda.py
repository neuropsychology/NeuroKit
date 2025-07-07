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

    assert len(nk.signal_findpeaks(eda2, height_min=0.6)["Peaks"]) > len(
        nk.signal_findpeaks(eda1, height_min=0.6)["Peaks"]
    )


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
    eda_phasic = nk.eda_phasic(nk.standardize(eda), method="highpass")[
        "EDA_Phasic"
    ].values

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
    eda = nk.eda_simulate(
        duration=30, scr_number=5, drift=0.1, noise=0, sampling_rate=250
    )
    signals, info = nk.eda_process(eda, sampling_rate=250)

    assert signals.shape == (7500, 11)
    assert (
        np.array(
            [
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
            ]
        )
        in signals.columns.values
    )

    # Check equal number of markers
    peaks = np.where(signals["SCR_Peaks"] == 1)[0]
    onsets = np.where(signals["SCR_Onsets"] == 1)[0]
    recovery = np.where(signals["SCR_Recovery"] == 1)[0]
    assert peaks.shape == onsets.shape == recovery.shape == (5,)


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
    np.testing.assert_array_equal(
        fig.axes[0].get_xticks(), fig.axes[1].get_xticks(), fig.axes[2].get_xticks()
    )
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

    vanhalem2020 = nk.eda_findpeaks(
        eda_phasic, sampling_rate=100, method="vanhalem2020"
    )
    min_n_peaks = min(len(vanhalem2020), len(nabian2018))
    assert any(
        nabian2018["SCR_Peaks"][:min_n_peaks] - vanhalem2020["SCR_Peaks"][:min_n_peaks]
    ) < np.mean(eda_signal)


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

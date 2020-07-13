import platform

import biosppy
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
        duration=30, sampling_rate=sampling_rate, scr_number=6, noise=0.01, drift=0.01, random_state=42
    )

    clean = nk.eda_clean(eda, sampling_rate=sampling_rate)
    assert len(clean) == len(eda)

    # Comparison to biosppy (https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/eda.py)

    eda_biosppy = nk.eda_clean(eda, sampling_rate=sampling_rate, method="biosppy")
    original, _, _ = biosppy.tools.filter_signal(
        signal=eda, ftype="butter", band="lowpass", order=4, frequency=5, sampling_rate=sampling_rate
    )

    original, _ = biosppy.tools.smoother(signal=original, kernel="boxzen", size=int(0.75 * sampling_rate), mirror=True)

    #    pd.DataFrame({"our":eda_biosppy, "biosppy":original}).plot()
    assert np.allclose((eda_biosppy - original).mean(), 0, atol=1e-5)


def test_eda_phasic():

    sampling_rate = 1000
    eda = nk.eda_simulate(
        duration=30, sampling_rate=sampling_rate, scr_number=6, noise=0.01, drift=0.01, random_state=42
    )

    if platform.system() == "Linux":
        cvxEDA = nk.eda_phasic(nk.standardize(eda), method="cvxeda")
        assert len(cvxEDA) == len(eda)

    smoothMedian = nk.eda_phasic(nk.standardize(eda), method="smoothmedian")
    assert len(smoothMedian) == len(eda)

    highpass = nk.eda_phasic(nk.standardize(eda), method="highpass")
    assert len(highpass) == len(eda)


def test_eda_peaks():

    sampling_rate = 1000
    eda = nk.eda_simulate(duration=30, sampling_rate=sampling_rate, scr_number=6, noise=0, drift=0.01, random_state=42)
    eda_phasic = nk.eda_phasic(nk.standardize(eda), method="highpass")["EDA_Phasic"].values

    signals, info = nk.eda_peaks(eda_phasic, method="gamboa2008")
    onsets, peaks, amplitudes = biosppy.eda.basic_scr(eda_phasic, sampling_rate=1000)
    assert np.allclose((info["SCR_Peaks"] - peaks).mean(), 0, atol=1e-5)

    signals, info = nk.eda_peaks(eda_phasic, method="kim2004")
    onsets, peaks, amplitudes = biosppy.eda.kbk_scr(eda_phasic, sampling_rate=1000)
    assert np.allclose((info["SCR_Peaks"] - peaks).mean(), 0, atol=1)


def test_eda_process():

    eda = nk.eda_simulate(duration=30, scr_number=5, drift=0.1, noise=0, sampling_rate=250)
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
    eda = nk.eda_simulate(duration=30, sampling_rate=sampling_rate, scr_number=6, noise=0, drift=0.01, random_state=42)
    eda_summary, _ = nk.eda_process(eda, sampling_rate=sampling_rate)

    # Plot data over samples.
    nk.eda_plot(eda_summary)
    # This will identify the latest figure.
    fig = plt.gcf()
    assert len(fig.axes) == 3
    titles = ["Raw and Cleaned Signal", "Skin Conductance Response (SCR)", "Skin Conductance Level (SCL)"]
    for (ax, title) in zip(fig.get_axes(), titles):
        assert ax.get_title() == title
    assert fig.get_axes()[2].get_xlabel() == "Samples"
    np.testing.assert_array_equal(fig.axes[0].get_xticks(), fig.axes[1].get_xticks(), fig.axes[2].get_xticks())
    plt.close(fig)

    # Plot data over seconds.
    nk.eda_plot(eda_summary, sampling_rate=sampling_rate)
    # This will identify the latest figure.
    fig = plt.gcf()
    assert fig.get_axes()[2].get_xlabel() == "Seconds"


def test_eda_eventrelated():

    eda = nk.eda_simulate(duration=15, scr_number=3)
    eda_signals, info = nk.eda_process(eda, sampling_rate=1000)
    epochs = nk.epochs_create(
        eda_signals, events=[5000, 10000, 15000], sampling_rate=1000, epochs_start=-0.1, epochs_end=1.9
    )
    eda_eventrelated = nk.eda_eventrelated(epochs)

    no_activation = np.where(eda_eventrelated["EDA_SCR"] == 0)[0][0]
    assert int(pd.DataFrame(eda_eventrelated.values[no_activation]).isna().sum()) == 4

    assert len(eda_eventrelated["Label"]) == 3

    # Test warning on missing columns
    with pytest.warns(nk.misc.NeuroKitWarning, match=r".*does not have an `EDA_Phasic`.*"):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["EDA_Phasic"]
        nk.eda_eventrelated({**epochs, first_epoch_key: first_epoch_copy})

    with pytest.warns(nk.misc.NeuroKitWarning, match=r".*does not have an `SCR_Amplitude`.*"):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["SCR_Amplitude"]
        nk.eda_eventrelated({**epochs, first_epoch_key: first_epoch_copy})

    with pytest.warns(nk.misc.NeuroKitWarning, match=r".*does not have an `SCR_RecoveryTime`.*"):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["SCR_RecoveryTime"]
        nk.eda_eventrelated({**epochs, first_epoch_key: first_epoch_copy})

    with pytest.warns(nk.misc.NeuroKitWarning, match=r".*does not have an `SCR_RiseTime`.*"):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["SCR_RiseTime"]
        nk.eda_eventrelated({**epochs, first_epoch_key: first_epoch_copy})


def test_eda_intervalrelated():

    data = nk.data("bio_resting_8min_100hz")
    df, info = nk.eda_process(data["EDA"], sampling_rate=100)
    columns = ["SCR_Peaks_N", "SCR_Peaks_Amplitude_Mean"]

    # Test with signal dataframe
    features_df = nk.eda_intervalrelated(df)

    assert all(elem in columns for elem in np.array(features_df.columns.values, dtype=str))
    assert features_df.shape[0] == 1  # Number of rows

    # Test with dict
    epochs = nk.epochs_create(df, events=[0, 25300], sampling_rate=100, epochs_end=20)
    features_dict = nk.eda_intervalrelated(epochs)

    assert all(elem in columns for elem in np.array(features_dict.columns.values, dtype=str))
    assert features_dict.shape[0] == 2  # Number of rows

import importlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import neurokit2 as nk


ecg_delineate_module = importlib.import_module("neurokit2.ecg.ecg_delineate")


def test_ecg_simulate():
    ecg1 = nk.ecg_simulate(duration=20, length=5000, method="simple", noise=0, random_state=0)
    assert len(ecg1) == 5000

    ecg2 = nk.ecg_simulate(duration=20, length=5000, heart_rate=500, random_state=1)
    #    pd.DataFrame({"ECG1":ecg1, "ECG2": ecg2}).plot()
    #    pd.DataFrame({"ECG1":ecg1, "ECG2": ecg2}).hist()
    assert len(nk.signal_findpeaks(ecg1, height_min=0.6)["Peaks"]) < len(nk.signal_findpeaks(ecg2, height_min=0.6)["Peaks"])

    ecg3 = nk.ecg_simulate(duration=10, length=5000, random_state=2)
    #    pd.DataFrame({"ECG1":ecg1, "ECG3": ecg3}).plot()
    assert len(nk.signal_findpeaks(ecg2, height_min=0.6)["Peaks"]) > len(nk.signal_findpeaks(ecg3, height_min=0.6)["Peaks"])


def test_ecg_clean():
    sampling_rate = 1000
    noise = 0.05

    ecg = nk.ecg_simulate(sampling_rate=sampling_rate, noise=noise, random_state=3)
    ecg_cleaned_nk = nk.ecg_clean(ecg, sampling_rate=sampling_rate, method="neurokit")

    assert ecg.size == ecg_cleaned_nk.size

    # Assert that highpass filter with .5 Hz lowcut was applied.
    fft_raw = np.abs(np.fft.rfft(ecg))
    fft_nk = np.abs(np.fft.rfft(ecg_cleaned_nk))

    freqs = np.fft.rfftfreq(ecg.size, 1 / sampling_rate)

    assert np.sum(fft_raw[freqs < 0.5]) > np.sum(fft_nk[freqs < 0.5])

    # Comparison to biosppy
    ecg_biosppy = nk.ecg_clean(ecg, sampling_rate=sampling_rate, method="biosppy")
    assert np.allclose(ecg_biosppy.mean(), 0, atol=1e-6)


def test_ecg_peaks():
    sampling_rate = 200
    noise = 1

    ecg = nk.ecg_simulate(duration=120, sampling_rate=sampling_rate, noise=noise, random_state=42)
    ecg[3000:3600] = 0

    # Test without request to correct artifacts.
    signals, _ = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit", correct_artifacts=False)

    assert signals.shape == (24000, 1)
    assert np.allclose(signals["ECG_R_Peaks"].values.sum(dtype=np.int64), 137, atol=1)

    # Test with request to correct artifacts.
    signals, info = nk.ecg_peaks(
        ecg,
        sampling_rate=sampling_rate,
        correct_artifacts=True,
        method="neurokit",
    )

    assert signals.shape == (24000, 1)
    assert np.allclose(signals["ECG_R_Peaks"].values.sum(dtype=np.int64), 136, atol=1)
    assert 17 in info["ECG_fixpeaks_longshort"]


def test_ecg_process():
    sampling_rate = 1000

    # Test 1: Normal ECG with noise
    ecg = nk.ecg_simulate(sampling_rate=sampling_rate, noise=0.05, random_state=4)
    signals, info = nk.ecg_process(ecg, sampling_rate=sampling_rate, method="neurokit")
    assert len(info["ECG_R_Peaks"]) > 0, "Should detect R-peaks in normal ECG"
    assert signals.shape[0] == len(ecg), "Signal length should match input"
    assert "ECG_Rate" in signals.columns, "Should have ECG_Rate column"
    assert "ECG_Quality" in signals.columns, "Should have ECG_Quality column"
    # When R-peaks are detected, results should NOT be NaN
    assert not np.all(np.isnan(signals["ECG_Rate"])), "ECG_Rate should not be all NaN when R-peaks are detected"
    assert not np.all(np.isnan(signals["ECG_Quality"])), "ECG_Quality should not be all NaN when R-peaks are detected"
    assert np.any(signals["ECG_R_Peaks"] == 1), "ECG_R_Peaks should contain some 1s when R-peaks are detected"

    # Test 2: Different heart rates
    for heart_rate in [50, 70, 120]:
        ecg = nk.ecg_simulate(duration=10, sampling_rate=sampling_rate, heart_rate=heart_rate, random_state=42)
        signals, info = nk.ecg_process(ecg, sampling_rate=sampling_rate)
        assert len(info["ECG_R_Peaks"]) > 0, f"Should detect R-peaks at {heart_rate} BPM"
        assert np.nanmean(signals["ECG_Rate"]) > 40, f"Heart rate should be reasonable at {heart_rate} BPM"
        # When R-peaks are detected, results should NOT be NaN
        assert not np.all(np.isnan(signals["ECG_Rate"])), f"ECG_Rate should not be all NaN at {heart_rate} BPM"
        assert not np.all(np.isnan(signals["ECG_Quality"])), f"ECG_Quality should not be all NaN at {heart_rate} BPM"

    # Test 3: Different noise levels
    for noise_level in [0.01, 0.05, 0.1]:
        ecg = nk.ecg_simulate(duration=10, sampling_rate=sampling_rate, noise=noise_level, random_state=42)
        signals, info = nk.ecg_process(ecg, sampling_rate=sampling_rate)
        assert len(info["ECG_R_Peaks"]) > 0, f"Should detect R-peaks with noise={noise_level}"
        # When R-peaks are detected, results should NOT be NaN
        assert not np.all(np.isnan(signals["ECG_Rate"])), f"ECG_Rate should not be all NaN with noise={noise_level}"
        assert not np.all(np.isnan(signals["ECG_Quality"])), f"ECG_Quality should not be all NaN with noise={noise_level}"

    # Test 4: Different methods
    for method in ["neurokit", "pantompkins1985"]:
        ecg = nk.ecg_simulate(duration=10, sampling_rate=sampling_rate, random_state=42)
        signals, info = nk.ecg_process(ecg, sampling_rate=sampling_rate, method=method)
        assert len(info["ECG_R_Peaks"]) > 0, f"Should detect R-peaks with method={method}"
        # When R-peaks are detected, results should NOT be NaN
        assert not np.all(np.isnan(signals["ECG_Rate"])), f"ECG_Rate should not be all NaN with method={method}"
        assert not np.all(np.isnan(signals["ECG_Quality"])), f"ECG_Quality should not be all NaN with method={method}"


def test_ecg_plot():
    ecg = nk.ecg_simulate(duration=60, heart_rate=70, noise=0.05, random_state=5)

    ecg_summary, info = nk.ecg_process(ecg, sampling_rate=1000, method="neurokit")

    # Plot data over seconds.
    nk.ecg_plot(ecg_summary, info)
    fig = plt.gcf()
    assert len(fig.axes) == 3
    assert fig.get_axes()[1].get_xlabel() == "Time (seconds)"
    np.testing.assert_array_equal(fig.axes[0].get_xticks(), fig.axes[1].get_xticks())
    plt.close(fig)

    # Make sure it works with cropped data
    nk.ecg_plot(ecg_summary[0:5000], info)
    fig = plt.gcf()
    assert fig.get_axes()[2].get_xlabel() == "Time (seconds)"


def test_ecg_findpeaks():
    sampling_rate = 1000

    ecg = nk.ecg_simulate(
        duration=60,
        sampling_rate=sampling_rate,
        noise=0,
        method="simple",
        random_state=42,
    )

    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=sampling_rate, method="neurokit")

    # Test neurokit methodwith show=True
    info_nk = nk.ecg_findpeaks(ecg_cleaned, show=True)

    assert info_nk["ECG_R_Peaks"].size == 69
    # This will identify the latest figure.
    fig = plt.gcf()
    assert len(fig.axes) == 2

    # Test pantompkins1985 method
    info_pantom = nk.ecg_findpeaks(nk.ecg_clean(ecg, method="pantompkins1985"), method="pantompkins1985")
    assert info_pantom["ECG_R_Peaks"].size == 70

    # Test hamilton2002 method
    info_hamilton = nk.ecg_findpeaks(nk.ecg_clean(ecg, method="hamilton2002"), method="hamilton2002")
    assert info_hamilton["ECG_R_Peaks"].size == 69

    # Test christov2004 method
    info_christov = nk.ecg_findpeaks(ecg_cleaned, method="christov2004")
    assert info_christov["ECG_R_Peaks"].size == 273

    # Test gamboa2008 method
    info_gamboa = nk.ecg_findpeaks(ecg_cleaned, method="gamboa2008")
    assert info_gamboa["ECG_R_Peaks"].size == 69

    # Test elgendi2010 method
    info_elgendi = nk.ecg_findpeaks(nk.ecg_clean(ecg, method="elgendi2010"), method="elgendi2010")
    assert info_elgendi["ECG_R_Peaks"].size == 70

    # Test engzeemod2012 method
    info_engzeemod = nk.ecg_findpeaks(nk.ecg_clean(ecg, method="engzeemod2012"), method="engzeemod2012")
    assert info_engzeemod["ECG_R_Peaks"].size == 69

    # Test kalidas2017 method
    info_kalidas = nk.ecg_findpeaks(nk.ecg_clean(ecg, method="kalidas2017"), method="kalidas2017")
    assert np.allclose(info_kalidas["ECG_R_Peaks"].size, 68, atol=1)

    # Test martinez2004 method
    ecg = nk.ecg_simulate(duration=60, sampling_rate=sampling_rate, noise=0, random_state=42)
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=sampling_rate, method="neurokit")
    info_martinez = nk.ecg_findpeaks(ecg_cleaned, method="martinez2004")
    assert np.allclose(info_martinez["ECG_R_Peaks"].size, 69, atol=1)

    # Test khamis2016 method
    info_khamis = nk.ecg_findpeaks(ecg_cleaned, method="khamis2016")
    assert info_khamis["ECG_R_Peaks"].size == 70


def test_ecg_eventrelated():
    ecg, _ = nk.ecg_process(nk.ecg_simulate(duration=20, random_state=6))
    epochs = nk.epochs_create(ecg, events=[5000, 10000, 15000], epochs_start=-0.1, epochs_end=1.9)
    ecg_eventrelated = nk.ecg_eventrelated(epochs)

    # Test rate features
    assert np.all(np.array(ecg_eventrelated["ECG_Rate_Min"]) < np.array(ecg_eventrelated["ECG_Rate_Mean"]))

    assert np.all(np.array(ecg_eventrelated["ECG_Rate_Mean"]) < np.array(ecg_eventrelated["ECG_Rate_Max"]))

    assert len(ecg_eventrelated["Label"]) == 3

    # Test warning on missing columns
    with pytest.warns(nk.misc.NeuroKitWarning, match=r".*does not have an `ECG_Phase_Artrial`.*"):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["ECG_Phase_Atrial"]
        nk.ecg_eventrelated({**epochs, first_epoch_key: first_epoch_copy})

    with pytest.warns(nk.misc.NeuroKitWarning, match=r".*does not have an.*`ECG_Phase_Ventricular`"):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["ECG_Phase_Ventricular"]
        nk.ecg_eventrelated({**epochs, first_epoch_key: first_epoch_copy})

    with pytest.warns(nk.misc.NeuroKitWarning, match=r".*does not have an `ECG_Quality`.*"):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["ECG_Quality"]
        nk.ecg_eventrelated({**epochs, first_epoch_key: first_epoch_copy})

    with pytest.warns(nk.misc.NeuroKitWarning, match=r".*does not have an `ECG_Rate`.*"):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        del first_epoch_copy["ECG_Rate"]
        nk.ecg_eventrelated({**epochs, first_epoch_key: first_epoch_copy})

    # Test warning on long epochs (eventrelated_utils)
    with pytest.warns(nk.misc.NeuroKitWarning, match=r".*duration of your epochs seems.*"):
        first_epoch_key = list(epochs.keys())[0]
        first_epoch_copy = epochs[first_epoch_key].copy()
        first_epoch_copy.index = range(len(first_epoch_copy))
        nk.ecg_eventrelated({**epochs, first_epoch_key: first_epoch_copy})


def test_ecg_delineate():
    sampling_rate = 1000

    # test with simulated signals
    ecg = nk.ecg_simulate(duration=20, sampling_rate=sampling_rate, random_state=42)
    _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
    number_rpeaks = len(rpeaks["ECG_R_Peaks"])

    # Method 1: derivative
    _, waves_derivative = nk.ecg_delineate(ecg, rpeaks, sampling_rate=sampling_rate, method="peaks")
    assert len(waves_derivative["ECG_P_Peaks"]) == number_rpeaks
    assert len(waves_derivative["ECG_Q_Peaks"]) == number_rpeaks
    assert len(waves_derivative["ECG_S_Peaks"]) == number_rpeaks
    assert len(waves_derivative["ECG_T_Peaks"]) == number_rpeaks
    assert len(waves_derivative["ECG_P_Onsets"]) == number_rpeaks
    assert len(waves_derivative["ECG_T_Offsets"]) == number_rpeaks

    # Method 2: CWT
    _, waves_cwt = nk.ecg_delineate(ecg, rpeaks, sampling_rate=sampling_rate, method="cwt")
    assert np.allclose(len(waves_cwt["ECG_P_Peaks"]), 22, atol=1)
    assert np.allclose(len(waves_cwt["ECG_T_Peaks"]), 22, atol=1)
    assert np.allclose(len(waves_cwt["ECG_R_Onsets"]), 23, atol=1)
    assert np.allclose(len(waves_cwt["ECG_R_Offsets"]), 23, atol=1)
    assert np.allclose(len(waves_cwt["ECG_P_Onsets"]), 22, atol=1)
    assert np.allclose(len(waves_cwt["ECG_P_Offsets"]), 22, atol=1)
    assert np.allclose(len(waves_cwt["ECG_T_Onsets"]), 22, atol=1)
    assert np.allclose(len(waves_cwt["ECG_T_Offsets"]), 22, atol=1)


def test_ecg_delineate_cwt_rpeaks_nonpositive_peak_height(monkeypatch):
    ecg = np.zeros(1000)
    rpeaks = np.array([250])

    def _fake_find_peaks(*args, **kwargs):
        return np.array([50]), {
            "peak_heights": np.array([0.0]),
            "left_bases": np.array([10]),
            "right_bases": np.array([10]),
        }

    monkeypatch.setattr(ecg_delineate_module.scipy.signal, "find_peaks", _fake_find_peaks)

    onsets, offsets = ecg_delineate_module._onset_offset_delineator(
        ecg,
        rpeaks,
        peak_type="rpeaks",
        sampling_rate=1000,
    )

    assert len(onsets) == 1
    assert len(offsets) == 1
    assert not np.isnan(onsets[0])
    assert not np.isnan(offsets[0])


def test_ecg_delineate_cwt_inverted_t_wave():
    """CWT delineation detects inverted T-waves at troughs, not spurious maxima (issue #1138)."""
    sampling_rate = 1000
    ecg = nk.ecg_simulate(duration=10, sampling_rate=sampling_rate, noise=0, random_state=42)
    _, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
    _, waves_upright = nk.ecg_delineate(ecg, info, sampling_rate=sampling_rate, method="cwt")

    # flip each T-wave window so T becomes a trough
    ecg_inv_t = ecg.copy()
    for t in [p for p in waves_upright["ECG_T_Peaks"] if not np.isnan(p)]:
        ecg_inv_t[int(t) - 50 : int(t) + 50] *= -1
    _, info_inv_t = nk.ecg_peaks(ecg_inv_t, sampling_rate=sampling_rate)
    _, waves = nk.ecg_delineate(ecg_inv_t, info_inv_t, sampling_rate=sampling_rate, method="cwt")

    t_peaks = [int(p) for p in waves["ECG_T_Peaks"] if not np.isnan(p)]
    assert len(t_peaks) > 5
    assert all(ecg_inv_t[p] < 0 for p in t_peaks), "Inverted T-peaks should be at troughs"

    # fully inverted ECG (aVR-like): T-peaks should still be at troughs
    ecg_inv = -ecg
    _, info_inv = nk.ecg_peaks(ecg_inv, sampling_rate=sampling_rate)
    _, waves = nk.ecg_delineate(ecg_inv, info_inv, sampling_rate=sampling_rate, method="cwt")
    t_peaks = [int(p) for p in waves["ECG_T_Peaks"] if not np.isnan(p)]
    assert len(t_peaks) > 5
    assert all(ecg_inv[p] < 0 for p in t_peaks), "Inverted T-peaks should be at troughs"


def test_ecg_invert():
    sampling_rate = 500
    noise = 0.05

    ecg = nk.ecg_simulate(sampling_rate=sampling_rate, noise=noise, random_state=3)
    ecg_inverted = ecg * -1 + 2 * np.nanmean(ecg)
    ecg_fixed, _ = nk.ecg_invert(ecg_inverted)
    assert np.allclose((ecg - ecg_fixed).mean(), 0, atol=1e-6)


def test_ecg_intervalrelated():
    data = nk.data("bio_resting_5min_100hz")
    df, _ = nk.ecg_process(data["ECG"], sampling_rate=100)

    columns = [
        "ECG_Rate_Mean",
        "HRV_RMSSD",
        "HRV_MeanNN",
        "HRV_SDNN",
        "HRV_SDSD",
        "HRV_CVNN",
        "HRV_CVSD",
        "HRV_MedianNN",
        "HRV_MadNN",
        "HRV_MCVNN",
        "HRV_IQRNN",
        "HRV_pNN50",
        "HRV_pNN20",
        "HRV_TINN",
        "HRV_HTI",
        "HRV_ULF",
        "HRV_VLF",
        "HRV_LF",
        "HRV_HF",
        "HRV_VHF",
        "HRV_LFHF",
        "HRV_LFn",
        "HRV_HFn",
        "HRV_LnHF",
        "HRV_SD1",
        "HRV_SD2",
        "HRV_SD1SD2",
        "HRV_S",
        "HRV_CSI",
        "HRV_CVI",
        "HRV_CSI_Modified",
        "HRV_PIP",
        "HRV_IALS",
        "HRV_PSS",
        "HRV_PAS",
        "HRV_GI",
        "HRV_SI",
        "HRV_AI",
        "HRV_PI",
        "HRV_C1d",
        "HRV_C1a",
        "HRV_SD1d",
        "HRV_SD1a",
        "HRV_C2d",
        "HRV_C2a",
        "HRV_SD2d",
        "HRV_SD2a",
        "HRV_Cd",
        "HRV_Ca",
        "HRV_SDNNd",
        "HRV_SDNNa",
        "HRV_DFA_alpha1",
        "HRV_DFA_alpha2",
        "HRV_ApEn",
        "HRV_SampEn",
        "HRV_MSE",
        "HRV_CMSE",
        "HRV_RCMSE",
        "HRV_CD",
    ]

    # Test with signal dataframe
    features_df = nk.ecg_intervalrelated(df, sampling_rate=100)

    # https://github.com/neuropsychology/NeuroKit/issues/304
    assert all(features_df == nk.ecg_analyze(df, sampling_rate=100, method="interval-related"))

    assert (elem in columns for elem in np.array(features_df.columns.values, dtype=str))
    assert features_df.shape[0] == 1  # Number of rows

    # Test with dict
    columns.append("Label")
    epochs = nk.epochs_create(df, events=[0, 15000], sampling_rate=100, epochs_end=150)
    features_dict = nk.ecg_intervalrelated(epochs, sampling_rate=100)

    assert (elem in columns for elem in np.array(features_dict.columns.values, dtype=str))
    assert features_dict.shape[0] == 2  # Number of rows


def test_ecg_process_no_rpeaks():
    """Test that ecg_process handles cases where no R-peaks are detected."""

    # Test with flatline ECG (no peaks)
    ecg_flat = np.zeros(1000)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        signals, info = nk.ecg_process(ecg_flat, sampling_rate=1000)

        # Check that no R-peaks were found
        assert len(info["ECG_R_Peaks"]) == 0, "Expected no R-peaks to be found"

        # Check that appropriate warnings were issued
        neurokit_warnings = [warning for warning in w if "No R-peaks were detected" in str(warning.message)]
        assert len(neurokit_warnings) >= 3, f"Expected at least 3 warnings, got {len(neurokit_warnings)}"

        # Check that signals DataFrame has expected structure
        assert signals.shape[0] == len(ecg_flat), "Signal length mismatch"
        assert "ECG_Rate" in signals.columns, "Missing ECG_Rate column"
        assert "ECG_Quality" in signals.columns, "Missing ECG_Quality column"

        # Check that rate and quality are NaN when no peaks are detected
        assert np.all(np.isnan(signals["ECG_Rate"])), "ECG_Rate should be NaN when no peaks detected"
        assert np.all(np.isnan(signals["ECG_Quality"])), "ECG_Quality should be NaN when no peaks detected"

        # Check that peak marker columns are zero-filled
        assert np.all(signals["ECG_R_Peaks"] == 0), "ECG_R_Peaks should be zero-filled when no peaks detected"


def test_ecg_delineate_no_rpeaks():
    """Test that ecg_delineate handles empty R-peaks array."""

    # Create flatline ECG and get empty R-peaks
    ecg_flat = np.zeros(1000)
    cleaned = nk.ecg_clean(ecg_flat, sampling_rate=1000)
    peaks, info = nk.ecg_peaks(cleaned, sampling_rate=1000)
    empty_rpeaks = info["ECG_R_Peaks"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test ecg_delineate with empty R-peaks
        signals, waves = nk.ecg_delineate(cleaned, rpeaks=empty_rpeaks, sampling_rate=1000)

        # Check that warning was issued
        delineate_warnings = [warning for warning in w if "delineation" in str(warning.message)]
        assert len(delineate_warnings) == 1, "ecg_delineate should issue one warning for empty R-peaks"

        # Check return values
        assert isinstance(signals, pd.DataFrame), "signals should be a DataFrame"
        assert isinstance(waves, dict), "waves should be a dict"
        assert signals.shape[0] == len(ecg_flat), "Signal length mismatch"
        assert len(waves["ECG_P_Peaks"]) == 0, "Waves should contain empty arrays"


def test_ecg_phase_no_rpeaks():
    """Test that ecg_phase handles empty R-peaks array."""

    # Create flatline ECG and get empty R-peaks
    ecg_flat = np.zeros(1000)
    cleaned = nk.ecg_clean(ecg_flat, sampling_rate=1000)
    peaks, info = nk.ecg_peaks(cleaned, sampling_rate=1000)
    empty_rpeaks = info["ECG_R_Peaks"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test ecg_phase with empty R-peaks
        phases = nk.ecg_phase(cleaned, rpeaks=empty_rpeaks, sampling_rate=1000)

        # Check that warning was issued
        phase_warnings = [warning for warning in w if "phase determination" in str(warning.message)]
        assert len(phase_warnings) == 1, "ecg_phase should issue one warning for empty R-peaks"

        # Check return values
        assert isinstance(phases, pd.DataFrame), "phases should be a DataFrame"
        assert phases.shape[0] == len(ecg_flat), "Phase length mismatch"
        assert np.all(np.isnan(phases["ECG_Phase_Atrial"])), "Phases should be NaN when no R-peaks"


def test_ecg_quality_no_rpeaks():
    """Test that ecg_quality handles empty R-peaks array."""

    # Create flatline ECG and get empty R-peaks
    ecg_flat = np.zeros(1000)
    cleaned = nk.ecg_clean(ecg_flat, sampling_rate=1000)
    peaks, info = nk.ecg_peaks(cleaned, sampling_rate=1000)
    empty_rpeaks = info["ECG_R_Peaks"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test ecg_quality with empty R-peaks
        quality = nk.ecg_quality(cleaned, rpeaks=empty_rpeaks, sampling_rate=1000)

        # Check that warning was issued
        quality_warnings = [warning for warning in w if "quality assessment" in str(warning.message)]
        assert len(quality_warnings) == 1, "ecg_quality should issue one warning for empty R-peaks"

        # Check return values
        assert isinstance(quality, np.ndarray), "quality should be a numpy array"
        assert len(quality) == len(ecg_flat), "Quality length mismatch"
        assert np.all(np.isnan(quality)), "Quality should be NaN when no R-peaks"

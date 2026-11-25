import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import neurokit2 as nk


SHOW_DEBUG_PLOTS = False
MAX_SIGNAL_DIFF = 0.03  # seconds


@pytest.fixture(name="test_data")
def setup_load_ecg_data():
    """Load ecg signal and sampling rate."""

    def load_signal_from_disk(filename=None, sampling_rate=2000):
        if filename is None:
            ecg = nk.ecg_simulate(duration=10, sampling_rate=sampling_rate, method="ecgsyn")
        else:
            filename = (pathlib.Path(__file__) / "../ecg_data" / filename).resolve().as_posix()
            ecg = np.array(pd.read_csv(filename))[:, 1]
        return ecg, sampling_rate

    ecg, sampling_rate = load_signal_from_disk("good_4000.csv", sampling_rate=4000)
    annots_filename = (pathlib.Path(__file__) / "../ecg_data" / "good_4000_annotation.csv").resolve().as_posix()
    annots = pd.read_csv(annots_filename, index_col=0, header=None).transpose()

    if SHOW_DEBUG_PLOTS:
        plt.plot(ecg)
        plt.show()

    rpeaks = nk.ecg_findpeaks(ecg, sampling_rate=sampling_rate, method="martinez")["ECG_R_Peaks"]
    test_data = dict(ecg=ecg, sampling_rate=sampling_rate, rpeaks=rpeaks)
    test_data.update(annots)
    yield test_data


def helper_plot(attribute, ecg_characteristics, test_data):
    peaks = [ecg_characteristics[attribute], test_data[attribute]]
    print("0: computed\n1: data")
    ecg = test_data["ecg"]
    nk.events_plot(peaks, ecg)
    plt.title(attribute)
    plt.show()


def run_test_func(test_data):
    _, waves = nk.ecg_delineate(test_data["ecg"], test_data["rpeaks"], test_data["sampling_rate"], method="dwt")
    for key in waves:
        waves[key] = np.array(waves[key])
    return waves


@pytest.mark.parametrize(
    "attribute",
    [
        "ECG_P_Peaks",
        "ECG_T_Peaks",
        "ECG_T_Onsets",
        "ECG_T_Offsets",
        "ECG_P_Onsets",
        "ECG_P_Offsets",
        "ECG_R_Onsets",
        "ECG_R_Offsets",
    ],
)
@pytest.mark.skipif(sys.platform.startswith("mac"), reason="skip testing on macos")
def test_find_ecg_characteristics(attribute, test_data):
    ecg_characteristics = run_test_func(test_data)
    diff = ecg_characteristics[attribute] - test_data[attribute]
    diff = diff[diff.abs() < 0.5 * test_data["sampling_rate"]]  # remove obvious failure
    test_data[attribute] = test_data[attribute][~np.isnan(test_data[attribute])]
    diff = diff[~np.isnan(diff)]
    report = f"""
Difference statistics
{diff.describe()}
Difference:
{diff}
"""
    # helper_plot(attribute, ecg_characteristics, test_data)
    assert diff.std() < 0.11 * test_data["sampling_rate"], report
    assert diff.mean() < 0.11 * test_data["sampling_rate"], report


@pytest.mark.parametrize("scale, offset", [(0.001, 0), (2, 0), (1000, 0), (1, 5.0), (250, -3.0)])
def test_ecg_delineate_dwt_is_invariant_to_signal_units(scale, offset):
    """The DWT delineation must not depend on the amplitude units (e.g., mV vs. uV) or the baseline of the ECG."""
    sampling_rate = 1000
    ecg = nk.ecg_clean(np.asarray(nk.data("ecg_1000hz")), sampling_rate=sampling_rate)
    rpeaks = nk.ecg_findpeaks(ecg, sampling_rate=sampling_rate, method="neurokit")["ECG_R_Peaks"]

    _, reference = nk.ecg_delineate(ecg, rpeaks, sampling_rate=sampling_rate, method="dwt")
    _, rescaled = nk.ecg_delineate(ecg * scale + offset, rpeaks, sampling_rate=sampling_rate, method="dwt")

    for key, expected in reference.items():
        np.testing.assert_array_equal(np.array(rescaled[key], dtype=float), np.array(expected, dtype=float), err_msg=key)


def test_dwt_normalize_window():
    from neurokit2.ecg.ecg_delineate import _dwt_normalize_window

    window = np.array([2.0, 4.0, 10.0, 6.0])  # median = 5, absolute deviations = [3, 1, 5, 1]
    expected = [0.6, 0.2, 1.0, 0.2]
    np.testing.assert_allclose(_dwt_normalize_window(window), expected)
    np.testing.assert_allclose(_dwt_normalize_window(window, weight=0.3), np.array(expected) * 0.3)
    # Invariant to scale and baseline offset, and to the polarity of the waves
    np.testing.assert_allclose(_dwt_normalize_window(window * 1000 + 7), expected)
    np.testing.assert_allclose(_dwt_normalize_window(-window), expected)
    # Degenerate windows must not produce NaN/inf
    np.testing.assert_array_equal(_dwt_normalize_window(np.full(5, 3.0)), np.zeros(5))
    assert len(_dwt_normalize_window([])) == 0


def test_dwt_tpeak_latency_penalty_is_symmetric():
    """A T-peak candidate earlier than expected must not be favoured over one equally late."""
    from neurokit2.ecg.ecg_delineate import _dwt_delineate_tp_peaks

    # At 60 bpm and 250 Hz: qrs_width = 0.13 s and rt_duration = 0.25 s
    rpeaks = np.arange(250, 2501, 250)
    search_start = rpeaks[0] + 16  # 0.5 * qrs_width after R
    expected = search_start + 46  # rt_duration - 0.5 * qrs_width into the search window
    early, late = expected - 25, expected + 25  # 0.1 s before and after

    # Each candidate is a +/- pair in the wavelet transform; the later one has the higher ECG amplitude
    dwtmatr = np.zeros((9, 3000))
    dwtmatr[3, [early, late]] = 1  # row 3 = degree_tpeak
    dwtmatr[3, [early + 2, late + 2]] = -1
    ecg = np.zeros(3000)
    ecg[early], ecg[late] = 0.8, 1.0

    tpeaks, _ = _dwt_delineate_tp_peaks(ecg, rpeaks, dwtmatr, sampling_rate=250)
    assert tpeaks[0] == late


def test_ecg_delineate_boundary_windows_and_cwt_cache(monkeypatch):
    """Boundary R-peaks near sample 0 or signal end must not fail or drop early waves."""
    import pywt

    from neurokit2.ecg.ecg_delineate import _find_tppeaks, _onset_offset_delineator, _peaks_delineator

    sampling_rate = 1000
    ecg = nk.ecg_simulate(duration=12, sampling_rate=sampling_rate, heart_rate=60, random_state=42)
    _, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
    rpeaks = info["ECG_R_Peaks"]

    # Slice so the first R-peak sits at ~300 ms (< 2 * p2r_duration = 400 ms) with its P-wave intact
    shift = max(0, int(rpeaks[1]) - 300)
    ecg_early_p = ecg[shift:]
    rpeaks_early_p = rpeaks[rpeaks >= shift] - shift
    _, waves_dwt = nk.ecg_delineate(ecg_early_p, rpeaks_early_p, sampling_rate=sampling_rate, method="dwt")
    assert not np.isnan(waves_dwt["ECG_P_Peaks"][0])
    assert not np.isnan(waves_dwt["ECG_P_Onsets"][0])
    assert not np.isnan(waves_dwt["ECG_R_Onsets"][0])

    # Slice so the first R-peak is within 40 ms of sample 0 (< half_wave_width = 100 samples)
    shift_very_early = int(rpeaks[1]) - 40
    ecg_edge = ecg[shift_very_early:]
    rpeaks_edge = rpeaks[rpeaks >= shift_very_early] - shift_very_early

    # Verify CWT computes pywt.cwt only once across the entire delineation and handles edge R-peaks
    cwt_calls = 0
    orig_cwt = pywt.cwt

    def counted_cwt(*args, **kwargs):
        nonlocal cwt_calls
        cwt_calls += 1
        return orig_cwt(*args, **kwargs)

    monkeypatch.setattr(pywt, "cwt", counted_cwt)
    signals_cwt, waves_cwt = nk.ecg_delineate(ecg_edge, rpeaks_edge, sampling_rate=sampling_rate, method="cwt")
    assert cwt_calls == 1
    assert len(waves_cwt["ECG_R_Onsets"]) == len(rpeaks_edge)
    assert signals_cwt.shape[0] == len(ecg_edge)

    # Edge cases in CWT helpers: single R-peak, closely spaced R-peaks, peak at 0, and early zero-crossing
    t_empty, p_empty = _peaks_delineator(ecg_edge, np.array([100]), sampling_rate=sampling_rate)
    assert len(t_empty) == 0 and len(p_empty) == 0
    t_close, p_close = _peaks_delineator(ecg_edge, np.array([100, 120]), sampling_rate=sampling_rate)
    assert np.isnan(t_close[0]) and np.isnan(p_close[0])
    on0, off0 = _onset_offset_delineator(ecg_edge, [0, len(ecg_edge)], peak_type="rpeaks", sampling_rate=sampling_rate)
    assert np.isnan(on0[0]) and np.isnan(off0[1])
    tp_early = _find_tppeaks(ecg_edge, [5, 25], sampling_rate=sampling_rate)
    assert len(tp_early) >= 1


def test_ecg_delineate_check_and_prominence_single_beat(monkeypatch):
    """check=True must remove >3 SD outliers in both waves and signals on pandas 3, including CWT."""
    import importlib

    delineate_mod = importlib.import_module("neurokit2.ecg.ecg_delineate")

    sampling_rate = 1000
    ecg = nk.ecg_simulate(duration=20, sampling_rate=sampling_rate, heart_rate=60, random_state=42)
    _, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
    rpeaks = info["ECG_R_Peaks"]

    # CWT with check=True (unequal P/T vs R feature lengths)
    signals_cwt, waves_cwt = nk.ecg_delineate(ecg, rpeaks, sampling_rate=sampling_rate, method="cwt", check=True)
    assert len(waves_cwt["ECG_P_Peaks"]) == len(rpeaks) - 1
    assert len(waves_cwt["ECG_R_Onsets"]) == len(rpeaks)
    assert signals_cwt.shape[0] == len(ecg)

    # Inject a >3 SD outlier and a valid index 0 + negative out-of-bounds index to verify sanitization & check=True
    orig_dwt = delineate_mod._dwt_ecg_delineator

    def fake_dwt(ecg_in, rpeaks_in, sampling_rate=1000):
        out = orig_dwt(ecg_in, rpeaks_in, sampling_rate=sampling_rate)
        out["ECG_P_Peaks"][5] = int(rpeaks_in[5] + 4500)  # >3 SD from R-peak
        out["ECG_P_Onsets"][0] = 0  # valid sample 0 must be preserved
        out["ECG_Q_Peaks"][0] = -5  # negative index must become NaN without shrinking list
        return out

    monkeypatch.setattr(delineate_mod, "_dwt_ecg_delineator", fake_dwt)
    outlier_idx = int(rpeaks[5] + 4500)
    signals_raw, waves_raw = nk.ecg_delineate(
        ecg.tolist(), pd.Series(rpeaks), sampling_rate=sampling_rate, method="dwt", check=False
    )
    assert len(waves_raw["ECG_Q_Peaks"]) == len(rpeaks)
    assert np.isnan(waves_raw["ECG_Q_Peaks"][0])
    assert waves_raw["ECG_P_Onsets"][0] == 0
    assert signals_raw["ECG_P_Onsets"].iloc[0] == 1

    signals_checked, waves_checked = nk.ecg_delineate(
        ecg.tolist(), pd.Series(rpeaks), sampling_rate=sampling_rate, method="dwt", check=True
    )
    assert np.isnan(waves_checked["ECG_P_Peaks"][5])
    assert signals_checked["ECG_P_Peaks"].iloc[outlier_idx] == 0
    assert np.isnan(waves_checked["ECG_P_Onsets"][0])
    assert signals_checked["ECG_P_Onsets"].iloc[0] == 0

    # Prominence method on a single beat (with check=True) and multi-epoch dict input
    signals_1, waves_1 = nk.ecg_delineate(
        ecg[:1200], rpeaks=[500], sampling_rate=sampling_rate, method="prominence", check=True
    )
    assert len(waves_1["ECG_R_Onsets"]) == 1
    assert signals_1.shape[0] == 1200

    df_clean = pd.DataFrame({"ECG_Clean": ecg})
    epochs = nk.epochs_create(df_clean, events=[1000, 6000], sampling_rate=sampling_rate, epochs_start=0, epochs_end=5)
    signals_ep, waves_ep = nk.ecg_delineate(epochs, sampling_rate=sampling_rate, method="prominence")
    assert len(signals_ep) == 2 * 5 * sampling_rate
    assert len(waves_ep["ECG_P_Peaks"]) > 0

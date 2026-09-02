import numpy as np
import pandas as pd
import pytest

import neurokit2 as nk
from neurokit2 import misc
from neurokit2.hrv.hrv_nonlinear import _hrv_symbolic_classify


def test_hrv_time():
    ecg_slow = nk.ecg_simulate(duration=60, sampling_rate=1000, heart_rate=60, random_state=42)
    ecg_fast = nk.ecg_simulate(duration=60, sampling_rate=1000, heart_rate=150, random_state=42)

    _, peaks_slow = nk.ecg_process(ecg_slow, sampling_rate=1000)
    _, peaks_fast = nk.ecg_process(ecg_fast, sampling_rate=1000)

    hrv_slow = nk.hrv_time(peaks_slow, sampling_rate=1000)
    hrv_fast = nk.hrv_time(peaks_fast, sampling_rate=1000)

    assert np.all(hrv_fast["HRV_RMSSD"] < hrv_slow["HRV_RMSSD"])
    assert np.all(hrv_fast["HRV_MeanNN"] < hrv_slow["HRV_MeanNN"])
    assert np.all(hrv_fast["HRV_SDNN"] < hrv_slow["HRV_SDNN"])
    assert np.all(hrv_fast["HRV_CVNN"] < hrv_slow["HRV_CVNN"])
    assert np.all(hrv_fast["HRV_CVSD"] < hrv_slow["HRV_CVSD"])
    assert np.all(hrv_fast["HRV_MedianNN"] < hrv_slow["HRV_MedianNN"])
    assert np.all(hrv_fast["HRV_MadNN"] < hrv_slow["HRV_MadNN"])
    assert np.all(hrv_fast["HRV_MCVNN"] < hrv_slow["HRV_MCVNN"])
    assert np.all(hrv_fast["HRV_pNN50"] == hrv_slow["HRV_pNN50"])
    assert np.all(hrv_fast["HRV_pNN20"] < hrv_slow["HRV_pNN20"])
    assert np.all(hrv_fast["HRV_TINN"] < hrv_slow["HRV_TINN"])
    assert np.all(hrv_fast["HRV_HTI"] != hrv_slow["HRV_HTI"])


def test_hrv_frequency():
    # Test frequency domain
    ecg1 = nk.ecg_simulate(duration=60, sampling_rate=2000, heart_rate=70, random_state=42)
    _, peaks1 = nk.ecg_process(ecg1, sampling_rate=2000)
    hrv1 = nk.hrv_frequency(peaks1, sampling_rate=2000)

    ecg2 = nk.signal_resample(ecg1, sampling_rate=2000, desired_sampling_rate=500)
    _, peaks2 = nk.ecg_process(ecg2, sampling_rate=500)
    hrv2 = nk.hrv_frequency(peaks2, sampling_rate=500)

    assert np.allclose(hrv1["HRV_HF"] - hrv2["HRV_HF"], 0, atol=1.5)
    assert np.isnan(hrv1["HRV_ULF"][0])
    assert np.isnan(hrv1["HRV_VLF"][0])
    assert np.isnan(hrv2["HRV_ULF"][0])
    assert np.isnan(hrv2["HRV_VLF"][0])


def test_hrv():
    ecg = nk.ecg_simulate(duration=120, sampling_rate=1000, heart_rate=110, random_state=42)

    _, peaks = nk.ecg_process(ecg, sampling_rate=1000)

    ecg_hrv = nk.hrv(peaks, sampling_rate=1000)

    assert np.isclose(ecg_hrv["HRV_RMSSD"].values[0], 3.526, atol=0.1)


def test_hrv_rqa():
    intervals = np.tile([900, 1000, 1100, 950, 1050], 8)
    peaks = np.r_[0, np.cumsum(intervals)]

    rqa = nk.hrv_rqa(peaks, sampling_rate=1000)

    assert rqa.shape == (1, 14)
    assert set(rqa.columns) == {
        "RecurrenceRate",
        "DiagRec",
        "Determinism",
        "DeteRec",
        "L",
        "Divergence",
        "LEn",
        "Laminarity",
        "TrappingTime",
        "VMax",
        "VEn",
        "W",
        "WMax",
        "WEn",
    }


def test_hrv_rqa_parameters():
    intervals = np.tile([900, 1000, 1100, 950, 1050], 8)
    peaks = np.r_[0, np.cumsum(intervals)]

    rqa_default = nk.hrv_rqa(peaks, sampling_rate=1000)
    rqa_custom = nk.hrv_rqa(peaks, sampling_rate=1000, dimension=3, delay=2, tolerance=10)

    assert not np.isclose(rqa_default["RecurrenceRate"].iloc[0], rqa_custom["RecurrenceRate"].iloc[0])


def test_hrv_rqa_short_input():
    with pytest.raises(ValueError, match=r"dimension \* delay"):
        nk.hrv_rqa([0, 1000, 2100], sampling_rate=1000)


def test_rri_input_hrv():
    ecg = nk.ecg_simulate(duration=120, sampling_rate=1000, heart_rate=110, random_state=42)

    _, peaks = nk.ecg_process(ecg, sampling_rate=1000)
    peaks = peaks["ECG_R_Peaks"]
    rri = np.diff(peaks).astype(float)
    rri_time = peaks[1:] / 1000

    rri[3:5] = [np.nan, np.nan]

    ecg_hrv = nk.hrv({"RRI": rri, "RRI_Time": rri_time})

    assert np.isclose(ecg_hrv["HRV_RMSSD"].values[0], 3.526, atol=0.2)


@pytest.mark.parametrize("detrend", ["polynomial", "loess"])
def test_hrv_detrended_rri(detrend):
    ecg = nk.ecg_simulate(duration=120, sampling_rate=1000, heart_rate=110, random_state=42)

    _, peaks = nk.ecg_process(ecg, sampling_rate=1000)
    peaks = peaks["ECG_R_Peaks"]
    rri = np.diff(peaks).astype(float)
    rri_time = peaks[1:] / 1000

    rri_processed, rri_processed_time, _ = nk.intervals_process(
        rri,
        intervals_time=rri_time,
        interpolate=False,
        interpolation_rate=None,
        detrend=detrend,
    )

    ecg_hrv = nk.hrv({"RRI": rri_processed, "RRI_Time": rri_processed_time})

    assert np.isclose(
        ecg_hrv["HRV_RMSSD"].values[0],
        np.sqrt(np.mean(np.square(np.diff(rri_processed)))),
        atol=0.1,
    )


@pytest.mark.parametrize("interpolation_rate", ["from_mean_rri", 1, 4, 10])
def test_hrv_interpolated_rri(interpolation_rate):
    ecg = nk.ecg_simulate(duration=120, sampling_rate=1000, heart_rate=110, random_state=42)

    _, peaks = nk.ecg_process(ecg, sampling_rate=1000)
    peaks = peaks["ECG_R_Peaks"]
    rri = np.diff(peaks).astype(float)
    rri_time = peaks[1:] / 1000

    if interpolation_rate == "from_mean_rri":
        interpolation_rate = 1000 / np.mean(rri)

    rri_processed, rri_processed_time, _ = nk.intervals_process(
        rri,
        intervals_time=rri_time,
        interpolate=True,
        interpolation_rate=interpolation_rate,
    )

    ecg_hrv = nk.hrv({"RRI": rri_processed, "RRI_Time": rri_processed_time})

    assert np.isclose(
        ecg_hrv["HRV_RMSSD"].values[0],
        np.sqrt(np.mean(np.square(np.diff(rri_processed)))),
        atol=0.1,
    )


def test_hrv_missing():
    random_state = 42
    rng = misc.check_random_state(random_state)
    # Download data
    data = nk.data("bio_resting_5min_100hz")
    sampling_rate = 100
    ecg = data["ECG"]

    _, peaks = nk.ecg_process(ecg, sampling_rate=sampling_rate)
    peaks = peaks["ECG_R_Peaks"]

    rri = np.diff(peaks / sampling_rate).astype(float) * 1000
    rri_time = peaks[1:] / sampling_rate

    # remove some intervals and their corresponding timestamps
    missing = rng.choice(len(rri), size=int(len(rri) / 5))
    rri_missing = rri[np.array([i for i in range(len(rri)) if i not in missing])]
    rri_time_missing = rri_time[np.array([i for i in range(len(rri_time)) if i not in missing])]

    orig_hrv = nk.hrv_time(peaks, sampling_rate=sampling_rate)
    miss_only_rri_hrv = nk.hrv_time({"RRI": rri_missing})
    # by providing the timestamps corresponding to each interval
    # we should be able to better estimate the original RMSSD
    # before some intervals were removed
    # (at least for this example signal)
    miss_rri_time_hrv = nk.hrv_time({"RRI": rri_missing, "RRI_Time": rri_time_missing})

    abs_diff_only_rri = np.mean(
        np.abs(
            np.diff(
                [
                    orig_hrv["HRV_RMSSD"].values[0],
                    miss_only_rri_hrv["HRV_RMSSD"].values[0],
                ]
            )
        )
    )
    abs_diff_rri_time = np.mean(
        np.abs(
            np.diff(
                [
                    orig_hrv["HRV_RMSSD"].values[0],
                    miss_rri_time_hrv["HRV_RMSSD"].values[0],
                ]
            )
        )
    )

    assert abs_diff_only_rri > abs_diff_rri_time


def test_hrv_rsa():
    data = nk.data("bio_eventrelated_100hz")
    ecg_signals, info = nk.ecg_process(data["ECG"], sampling_rate=100)
    rsp_signals, _ = nk.rsp_process(data["RSP"], sampling_rate=100)

    rsa_feature_columns = [
        "RSA_P2T_Mean",
        "RSA_P2T_Mean_log",
        "RSA_P2T_SD",
        "RSA_P2T_NoRSA",
        "RSA_PorgesBohrer",
        "RSA_Gates_Mean",
        "RSA_Gates_Mean_log",
        "RSA_Gates_SD",
    ]

    rsa_features = nk.hrv_rsa(ecg_signals, rsp_signals, rpeaks=info, sampling_rate=100, continuous=False)

    assert all(key in rsa_feature_columns for key in rsa_features.keys())

    # Test simulate RSP signal warning
    with pytest.warns(misc.NeuroKitWarning, match=r"RSP signal not found. For this.*"):
        nk.hrv_rsa(ecg_signals, rpeaks=info, sampling_rate=100, continuous=False)

    with pytest.warns(misc.NeuroKitWarning, match=r"RSP signal not found. For this time.*"):
        nk.hrv_rsa(
            ecg_signals,
            pd.DataFrame(),
            rpeaks=info,
            sampling_rate=100,
            continuous=False,
        )

    # Test missing rsp onsets/centers
    with pytest.warns(misc.NeuroKitWarning, match=r"Couldn't find rsp cycles onsets and centers.*"):
        rsp_signals["RSP_Peaks"] = 0
        _ = nk.hrv_rsa(ecg_signals, rsp_signals, rpeaks=info, sampling_rate=100, continuous=False)


def test_hrv_nonlinear_fragmentation():
    # https://github.com/neuropsychology/NeuroKit/issues/344
    from neurokit2.hrv.hrv_nonlinear import _hrv_nonlinear_fragmentation

    edge_rri = np.array([888.0, 1262.0, 1290.0, 1274.0, 1300.0, 1244.0, 1266.0])
    test_out = {}

    _hrv_nonlinear_fragmentation(edge_rri, out=test_out)
    assert test_out == {
        "IALS": 0.8333333333333334,
        "PAS": 1.0,
        "PIP": 0.5714285714285714,
        "PSS": 1.0,
    }


def test_hrv_symbolic_classify():
    """Unit test for _hrv_symbolic_classify — verifies the bug-fix that [a,b,a] is 2UV not 1V."""
    a, b = 0, 1

    # 0V: all equal
    result = _hrv_symbolic_classify(np.array([[a, a, a]]))
    assert result["0V"] == 1.0
    assert result["1V"] == result["2LV"] == result["2UV"] == 0.0

    # 1V: one adjacent change (a==b, b!=c)
    result = _hrv_symbolic_classify(np.array([[a, a, b]]))
    assert result["1V"] == 1.0

    # 1V: one adjacent change (a!=b, b==c)
    result = _hrv_symbolic_classify(np.array([[a, b, b]]))
    assert result["1V"] == 1.0

    # 2UV: [a, b, a] — two changes, non-monotone (peak/valley); the critical bug case
    result = _hrv_symbolic_classify(np.array([[a, b, a]]))
    assert result["2UV"] == 1.0, "[a,b,a] must be 2UV, not 1V"
    assert result["1V"] == 0.0

    # 2LV: monotone increasing [0, 1, 2]
    result = _hrv_symbolic_classify(np.array([[0, 1, 2]]))
    assert result["2LV"] == 1.0

    # 2LV: monotone decreasing [2, 1, 0]
    result = _hrv_symbolic_classify(np.array([[2, 1, 0]]))
    assert result["2LV"] == 1.0

    # 2UV: non-monotone with two changes [0, 2, 1]
    result = _hrv_symbolic_classify(np.array([[0, 2, 1]]))
    assert result["2UV"] == 1.0


def test_hrv_symbolic():
    """Integration test: correct columns, values in [0,1], each family group sums to 1."""
    ecg = nk.ecg_simulate(duration=60, sampling_rate=250, heart_rate=70, random_state=42)
    _, peaks = nk.ecg_process(ecg, sampling_rate=250)
    peaks = peaks["ECG_R_Peaks"]

    HRV_Symbolic = nk.hrv_symbolic(peaks)

    expected_columns = [
        "HRV_Symbolic_EqualProb4_0V",
        "HRV_Symbolic_EqualProb4_1V",
        "HRV_Symbolic_EqualProb4_2LV",
        "HRV_Symbolic_EqualProb4_2UV",
    ]
    assert list(HRV_Symbolic.columns) == expected_columns

    values = HRV_Symbolic.iloc[0].values
    assert np.all(values >= 0.0) and np.all(values <= 1.0)

    # Each group of 4 families (0V+1V+2LV+2UV) must sum to ~1.0
    for start in range(0, len(values), 4):
        group_sum = values[start : start + 4].sum()
        assert np.isclose(group_sum, 1.0, atol=1e-9), f"Family group at col {start} sums to {group_sum}"

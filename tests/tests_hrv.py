import numpy as np
import pytest

import neurokit2 as nk


def test_hrv_time():
    ecg_slow = nk.ecg_simulate(duration=60, sampling_rate=1000, heart_rate=70, random_state=42)
    ecg_fast = nk.ecg_simulate(duration=60, sampling_rate=1000, heart_rate=110, random_state=42)

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
    assert np.all(hrv_fast["HRV_HTI"] > hrv_slow["HRV_HTI"])


def test_hrv_frequency():
    # Test frequency domain
    ecg1 = nk.ecg_simulate(duration=60, sampling_rate=2000, heart_rate=70, random_state=42)
    _, peaks1 = nk.ecg_process(ecg1, sampling_rate=2000)
    hrv1 = nk.hrv_frequency(peaks1, sampling_rate=2000)

    ecg2 = nk.signal_resample(ecg1, sampling_rate=2000, desired_sampling_rate=500)
    _, peaks2 = nk.ecg_process(ecg2, sampling_rate=500)
    hrv2 = nk.hrv_frequency(peaks2, sampling_rate=500)

    assert np.allclose(hrv1["HRV_HF"] - hrv2["HRV_HF"], 0, atol=1.5)
    assert np.isnan(hrv1["HRV_LF"][0])
    assert np.isnan(hrv2["HRV_LF"][0])
    assert np.isnan(hrv1["HRV_VLF"][0])
    assert np.isnan(hrv2["HRV_LF"][0])

    # Test warning on too short duration
    with pytest.warns(nk.misc.NeuroKitWarning, match=r"The duration of recording is too short.*"):
        ecg3 = nk.ecg_simulate(duration=10, sampling_rate=2000, heart_rate=70, random_state=42)
        _, peaks3 = nk.ecg_process(ecg3, sampling_rate=2000)
        nk.hrv_frequency(peaks3, sampling_rate=2000, silent=False)


def test_hrv():

    ecg = nk.ecg_simulate(duration=60, sampling_rate=1000, heart_rate=110, random_state=42)

    _, peaks = nk.ecg_process(ecg, sampling_rate=1000)

    ecg_hrv = nk.hrv(peaks, sampling_rate=1000)

    columns = ['HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDSD', 'HRV_CVNN',
               'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN',
               'HRV_pNN50', 'HRV_pNN20', 'HRV_TINN', 'HRV_HTI', 'HRV_ULF',
               'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF', 'HRV_LFn',
               'HRV_HFn', 'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S',
               'HRV_CSI', 'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS',
               'HRV_PSS', 'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI',
               'HRV_C1d', 'HRV_C1a', 'HRV_SD1d',
               'HRV_SD1a', 'HRV_C2d',
               'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a',
               'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd',
               'HRV_SDNNa', 'HRV_ApEn', 'HRV_SampEn']

    assert all(elem in np.array(ecg_hrv.columns.values, dtype=object) for elem
               in columns)


def test_hrv_rsa():
    data = nk.data("bio_eventrelated_100hz")
    ecg_signals, info = nk.ecg_process(data["ECG"], sampling_rate=100)
    rsp_signals, _ = nk.rsp_process(data["RSP"], sampling_rate=100)

    rsa_feature_columns = [
      'RSA_P2T_Mean',
      'RSA_P2T_Mean_log',
      'RSA_P2T_SD',
      'RSA_P2T_NoRSA',
      'RSA_PorgesBohrer',
     ]

    rsa_features = nk.hrv_rsa(
        ecg_signals,
        rsp_signals,
        rpeaks=info,
        sampling_rate=100,
        continuous=False
    )

    assert all(key in rsa_feature_columns for key in rsa_features.keys())

    # Test simulate RSP signal warning
    with pytest.warns(nk.misc.NeuroKitWarning, match=r"RSP signal not found.*"):
        # TODO: Traceback (most recent call last):
        #   File "main.py", line 11, in <module>
        #     nk.hrv_rsa(ecg_signals, rpeaks=info, sampling_rate=100, continuous=False)
        #   File "/home/alexander/sandbox/src/github.com/awwong1/NeuroKit/neurokit2/hrv/hrv_rsa.py", line 132, in hrv_rsa
        #     signals, ecg_period, rpeaks, __ = _hrv_rsa_formatinput(ecg_signals, rsp_signals, rpeaks, sampling_rate)
        #   File "/home/alexander/sandbox/src/github.com/awwong1/NeuroKit/neurokit2/hrv/hrv_rsa.py", line 345, in _hrv_rsa_formatinput
        #     edr = ecg_rsp(ecg_period, sampling_rate=sampling_rate)
        # TypeError: 'module' object is not callable

        nk.hrv_rsa(ecg_signals, rpeaks=info, sampling_rate=100, continuous=False)

# -*- coding: utf-8 -*-
import biosppy
import matplotlib.pyplot as plt
import numpy as np

import neurokit2 as nk


def test_ecg_simulate():

    ecg1 = nk.ecg_simulate(duration=20, length=5000, method="simple", noise=0)
    assert len(ecg1) == 5000

    ecg2 = nk.ecg_simulate(duration=20, length=5000, heart_rate=500)
#    pd.DataFrame({"ECG1":ecg1, "ECG2": ecg2}).plot()
#    pd.DataFrame({"ECG1":ecg1, "ECG2": ecg2}).hist()
    assert (len(nk.signal_findpeaks(ecg1, height_min=0.6)["Peaks"]) <
            len(nk.signal_findpeaks(ecg2, height_min=0.6)["Peaks"]))

    ecg3 = nk.ecg_simulate(duration=10, length=5000)
#    pd.DataFrame({"ECG1":ecg1, "ECG3": ecg3}).plot()
    assert (len(nk.signal_findpeaks(ecg2, height_min=0.6)["Peaks"]) >
            len(nk.signal_findpeaks(ecg3, height_min=0.6)["Peaks"]))


def test_ecg_clean():

    sampling_rate = 1000
    noise = 0.05

    ecg = nk.ecg_simulate(sampling_rate=sampling_rate, noise=noise)
    ecg_cleaned_nk = nk.ecg_clean(ecg, sampling_rate=sampling_rate,
                                  method="neurokit")

    assert ecg.size == ecg_cleaned_nk.size

    # Assert that highpass filter with .5 Hz lowcut was applied.
    fft_raw = np.abs(np.fft.rfft(ecg))
    fft_nk = np.abs(np.fft.rfft(ecg_cleaned_nk))

    freqs = np.fft.rfftfreq(ecg.size, 1 / sampling_rate)

    assert np.sum(fft_raw[freqs < .5]) > np.sum(fft_nk[freqs < .5])

    # Comparison to biosppy (https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L69)
    ecg_biosppy = nk.ecg_clean(ecg, sampling_rate=sampling_rate,
                                method="biosppy")
    original, _, _ = biosppy.tools.filter_signal(signal=ecg,
                                                  ftype='FIR',
                                                  band='bandpass',
                                                  order=int(0.3 * sampling_rate),
                                                  frequency=[3, 45],
                                                  sampling_rate=sampling_rate)
    assert np.allclose((ecg_biosppy - original).mean(), 0, atol=1e-6)


def test_ecg_peaks():

    sampling_rate = 1000
    noise = 0.15

    ecg = nk.ecg_simulate(duration=120, sampling_rate=sampling_rate,
                          noise=noise, random_state=42)
    ecg_cleaned_nk = nk.ecg_clean(ecg, sampling_rate=sampling_rate,
                                  method="neurokit")

    # Test without request to correct artifacts.
    signals, info = nk.ecg_peaks(ecg_cleaned_nk, correct_artifacts=False,
                                 method="neurokit")

    assert signals.shape == (120000, 1)
    assert np.allclose(signals["ECG_R_Peaks"].values.sum(dtype=np.int64), 139,
                       atol=1)

    # Test with request to correct artifacts.
    signals, info = nk.ecg_peaks(ecg_cleaned_nk,
                                 correct_artifacts=True,
                                 method="neurokit")

    assert signals.shape == (120000, 1)
    assert np.allclose(signals["ECG_R_Peaks"].values.sum(dtype=np.int64), 139,
                       atol=1)


def test_ecg_process():

    sampling_rate = 1000
    noise = 0.05

    ecg = nk.ecg_simulate(sampling_rate=sampling_rate, noise=noise)
    signals, info = nk.ecg_process(ecg, sampling_rate=sampling_rate,
                                   method="neurokit")



def test_ecg_plot():

    ecg = nk.ecg_simulate(duration=60, heart_rate=70, noise=0.05)

    ecg_summary, _ = nk.ecg_process(ecg, sampling_rate=1000, method="neurokit")

    # Plot data over samples.
    nk.ecg_plot(ecg_summary)
    # This will identify the latest figure.
    fig = plt.gcf()
    assert len(fig.axes) == 2
    titles = ["Raw and Cleaned Signal",
              "Heart Rate"]
    for (ax, title) in zip(fig.get_axes(), titles):
        assert ax.get_title() == title
    assert fig.get_axes()[1].get_xlabel() == "Samples"
    np.testing.assert_array_equal(fig.axes[0].get_xticks(),
                                  fig.axes[1].get_xticks())
    plt.close(fig)

    # Plot data over seconds.
    nk.ecg_plot(ecg_summary, sampling_rate=1000)
    # This will identify the latest figure.
    fig = plt.gcf()
    assert len(fig.axes) == 3
    titles = ["Raw and Cleaned Signal",
              "Heart Rate",
              "Individual Heart Beats"]
    for (ax, title) in zip(fig.get_axes(), titles):
        assert ax.get_title() == title
    assert fig.get_axes()[1].get_xlabel() == "Time (seconds)"
    np.testing.assert_array_equal(fig.axes[0].get_xticks(),
                                  fig.axes[1].get_xticks())
    plt.close(fig)


def test_ecg_findpeaks():

    sampling_rate = 1000

    ecg = nk.ecg_simulate(duration=60, sampling_rate=sampling_rate, noise=0,
                          method="simple", random_state=42)

    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=sampling_rate,
                               method="neurokit")

    # Test neurokit methodwith show=True
    info_nk = nk.ecg_findpeaks(ecg_cleaned, show=True)

    assert info_nk["ECG_R_Peaks"].size == 69
    # This will identify the latest figure.
    fig = plt.gcf()
    assert len(fig.axes) == 2

    # Test pantompkins1985 method
    info_pantom = nk.ecg_findpeaks(nk.ecg_clean(ecg,
                                   method="pantompkins1985"),
                                   method="pantompkins1985")
    assert info_pantom["ECG_R_Peaks"].size == 70

    # Test hamilton2002 method
    info_hamilton = nk.ecg_findpeaks(nk.ecg_clean(ecg, method="hamilton2002"),
                                   method="hamilton2002")
    assert info_hamilton["ECG_R_Peaks"].size == 69

    # Test christov2004 method
    info_christov = nk.ecg_findpeaks(ecg_cleaned, method="christov2004")
    assert info_christov["ECG_R_Peaks"].size == 273

    # Test gamboa2008 method
    info_gamboa = nk.ecg_findpeaks(ecg_cleaned, method="gamboa2008")
    assert info_gamboa["ECG_R_Peaks"].size == 69

    # Test elgendi2010 method
    info_elgendi = nk.ecg_findpeaks(nk.ecg_clean(ecg, method="elgendi2010"),
                                   method="elgendi2010")
    assert info_elgendi["ECG_R_Peaks"].size == 70

    # Test engzeemod2012 method
    info_engzeemod = nk.ecg_findpeaks(nk.ecg_clean(ecg,
                                    method="engzeemod2012"),
                                    method="engzeemod2012")
    assert info_engzeemod["ECG_R_Peaks"].size == 70

    # Test kalidas2017 method
    info_kalidas = nk.ecg_findpeaks(nk.ecg_clean(ecg, method="kalidas2017"),
                                      method="kalidas2017")
    assert info_kalidas["ECG_R_Peaks"].size == 69

    # Test martinez2003 method
    ecg = nk.ecg_simulate(duration=60, sampling_rate=sampling_rate, noise=0,
                          random_state=42)
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=sampling_rate,
                               method="neurokit")
    info_martinez = nk.ecg_findpeaks(ecg_cleaned, method="martinez2003")
    assert np.allclose(info_martinez["ECG_R_Peaks"].size,
                       69, atol=1)


def test_ecg_eventrelated():

    ecg, info = nk.ecg_process(nk.ecg_simulate(duration=20))
    epochs = nk.epochs_create(ecg, events=[5000, 10000, 15000],
                              epochs_start=-0.1, epochs_end=1.9)
    ecg_eventrelated = nk.ecg_eventrelated(epochs)

    # Test rate features
    assert np.alltrue(np.array(ecg_eventrelated["ECG_Rate_Min"]) <
                      np.array(ecg_eventrelated["ECG_Rate_Mean"]))

    assert np.alltrue(np.array(ecg_eventrelated["ECG_Rate_Mean"]) <
                      np.array(ecg_eventrelated["ECG_Rate_Max"]))

    assert len(ecg_eventrelated["Label"]) == 3


def test_ecg_delineate():

    sampling_rate = 1000

    # test with simulated signals
    ecg = nk.ecg_simulate(duration=20, sampling_rate=sampling_rate, random_state=42)
    _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
    number_rpeaks = len(rpeaks['ECG_R_Peaks'])

    # Method 1: derivative
    _, waves_derivative = nk.ecg_delineate(ecg, rpeaks, sampling_rate=sampling_rate)
    assert len(waves_derivative['ECG_P_Peaks']) == number_rpeaks
    assert len(waves_derivative['ECG_Q_Peaks']) == number_rpeaks
    assert len(waves_derivative['ECG_S_Peaks']) == number_rpeaks
    assert len(waves_derivative['ECG_T_Peaks']) == number_rpeaks
    assert len(waves_derivative['ECG_P_Onsets']) == number_rpeaks
    assert len(waves_derivative['ECG_T_Offsets']) == number_rpeaks

    # Method 2: CWT
    _, waves_cwt = nk.ecg_delineate(ecg, rpeaks, sampling_rate=sampling_rate, method='cwt')
    assert np.allclose(len(waves_cwt['ECG_P_Peaks']), 22, atol=1)
    assert np.allclose(len(waves_cwt['ECG_T_Peaks']), 22, atol=1)
    assert np.allclose(len(waves_cwt['ECG_R_Onsets']), 23, atol=1)
    assert np.allclose(len(waves_cwt['ECG_R_Offsets']), 23, atol=1)
    assert np.allclose(len(waves_cwt['ECG_P_Onsets']), 22, atol=1)
    assert np.allclose(len(waves_cwt['ECG_P_Offsets']), 22, atol=1)
    assert np.allclose(len(waves_cwt['ECG_T_Onsets']), 22, atol=1)
    assert np.allclose(len(waves_cwt['ECG_T_Offsets']), 22, atol=1)


def test_ecg_intervalrelated():

    data = nk.data("bio_resting_5min_100hz")
    df, info = nk.ecg_process(data["ECG"], sampling_rate=100)
    columns = ['ECG_Rate_Mean', 'HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDSD',
               'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN',
               'HRV_pNN50', 'HRV_pNN20', 'HRV_TINN', 'HRV_HTI', 'HRV_ULF',
               'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF', 'HRV_LFn',
               'HRV_HFn', 'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD2SD1',
               'HRV_CSI', 'HRV_CVI', 'HRV_CSI_Modified', 'HRV_SampEn']

    # Test with signal dataframe
    features_df = nk.ecg_intervalrelated(df)

    assert all(elem in columns for elem
               in np.array(features_df.columns.values, dtype=str))
    assert features_df.shape[0] == 1  # Number of rows

    # Test with dict
    epochs = nk.epochs_create(df, events=[0, 15000],
                              sampling_rate=100, epochs_end=150)
    features_dict = nk.ecg_intervalrelated(epochs, sampling_rate=100)

    assert all(elem in columns for elem
               in np.array(features_dict.columns.values, dtype=str))
    assert features_dict.shape[0] == 2  # Number of rows

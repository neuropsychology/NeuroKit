# -*- coding: utf-8 -*-
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt


def test_ecg_simulate():

    ecg1 = nk.ecg_simulate(duration=20, length=5000, method="simple", noise=0)
    assert len(ecg1) == 5000

    ecg2 = nk.ecg_simulate(duration=20, length=5000, heart_rate=500)
#    pd.DataFrame({"ECG1":ecg1, "ECG2": ecg2}).plot()
#    pd.DataFrame({"ECG1":ecg1, "ECG2": ecg2}).hist()
    assert (len(nk.signal_findpeaks(ecg1, height_min=0.6)[0]) <
            len(nk.signal_findpeaks(ecg2, height_min=0.6)[0]))

    ecg3 = nk.ecg_simulate(duration=10, length=5000)
#    pd.DataFrame({"ECG1":ecg1, "ECG3": ecg3}).plot()
    assert (len(nk.signal_findpeaks(ecg2, height_min=0.6)[0]) >
            len(nk.signal_findpeaks(ecg3, height_min=0.6)[0]))


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


def test_ecg_findpeaks():

    sampling_rate = 1000
    noise = 0.05

    ecg = nk.ecg_simulate(sampling_rate=sampling_rate, noise=noise)
    ecg_cleaned_nk = nk.ecg_clean(ecg, sampling_rate=sampling_rate,
                                  method="neurokit")
    signals, info = nk.ecg_findpeaks(ecg_cleaned_nk, method="neurokit")

    assert signals.shape == (10000, 1)
    assert np.allclose(signals["ECG_Peaks"].values.sum(dtype=np.int64), 11)
    assert info["ECG_Peaks"].shape[0] == 11
    assert np.allclose(info["ECG_Peaks"].sum(dtype=np.int64), 56552, atol=1)


def test_ecg_rate():

    sampling_rate = 1000
    noise = 0.05

    ecg = nk.ecg_simulate(sampling_rate=sampling_rate, noise=noise)
    ecg_cleaned_nk = nk.ecg_clean(ecg, sampling_rate=sampling_rate,
                                  method="neurokit")
    signals, info = nk.ecg_findpeaks(ecg_cleaned_nk, method="neurokit")

    # Test with dictionary.
    test_length = 30
    data = nk.ecg_rate(peaks=info, sampling_rate=sampling_rate,
                       desired_length=test_length)

    assert data.shape == (test_length, 1)
    assert np.abs(data["ECG_Rate"].mean() - 70.5) < 0.1

    # Test with DataFrame.
    data = nk.ecg_rate(peaks=signals, sampling_rate=sampling_rate)
    assert data.shape == (ecg.size, 1)
    assert np.abs(data["ECG_Rate"].mean() - 70.5) < 0.2


def test_ecg_process():

    sampling_rate = 1000
    noise = 0.05

    ecg = nk.ecg_simulate(sampling_rate=sampling_rate, noise=noise)
    signals, info = nk.ecg_process(ecg, sampling_rate=sampling_rate,
                                   method="neurokit")
    # Only check array dimensions and column names since functions called by
    # ecg_process have already been unit tested
    assert signals.shape == (10000, 4)
    for i in zip(signals.columns,
                 ["ECG_Raw", "ECG_Clean", "ECG_Peaks", "ECG_Rate"]):
        assert i[0] == i[1]


def test_ecg_plot():

    ecg = nk.ecg_simulate(duration=60, heart_rate=70, noise=0.05)

    ecg_summary, _ = nk.ecg_process(ecg, sampling_rate=1000, method="neurokit")

    # Plot data over samples.
    nk.ecg_plot(ecg_summary)
    # This will identify the latest figure.
    fig = plt.gcf()
    assert len(fig.axes) == 2
    titles = ["Raw and Cleaned ECG with R-peaks",
              "Heart Rate"]
    for (ax, title) in zip(fig.get_axes(), titles):
        assert ax.get_title() == title
    assert fig.get_axes()[1].get_xlabel() == "Samples"
    plt.close(fig)

    # Plot data over seconds.
    nk.ecg_plot(ecg_summary, sampling_rate=1000)
    # This will identify the latest figure.
    fig = plt.gcf()
    assert len(fig.axes) == 2
    titles = ["Raw and Cleaned ECG with R-peaks",
              "Heart Rate"]
    for (ax, title) in zip(fig.get_axes(), titles):
        assert ax.get_title() == title
    assert fig.get_axes()[1].get_xlabel() == "Time (seconds)"
    plt.close(fig)

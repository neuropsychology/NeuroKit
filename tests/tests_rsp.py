# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import neurokit2 as nk

import matplotlib.pyplot as plt
import scipy.stats


def test_rsp_simulate():
    rsp1 = nk.rsp_simulate(duration=20, length=3000)
    assert len(rsp1) == 3000

    rsp2 = nk.rsp_simulate(duration=20, length=3000, respiratory_rate=80)
#    pd.DataFrame({"RSP1":rsp1, "RSP2":rsp2}).plot()
#    pd.DataFrame({"RSP1":rsp1, "RSP2":rsp2}).hist()
    assert (len(nk.signal_findpeaks(rsp1, height_min=0.2)[0]) <
            len(nk.signal_findpeaks(rsp2, height_min=0.2)[0]))

    rsp3 = nk.rsp_simulate(duration=20, length=3000, method="sinusoidal")
    rsp4 = nk.rsp_simulate(duration=20, length=3000, method="breathmetrics")
#    pd.DataFrame({"RSP3":rsp3, "RSP4":rsp4}).plot()
    assert (len(nk.signal_findpeaks(rsp3, height_min=0.2)[0]) >
            len(nk.signal_findpeaks(rsp4, height_min=0.2)[0]))


def test_rsp_clean():

    sampling_rate = 1000
    rsp = nk.rsp_simulate(duration=120, sampling_rate=sampling_rate,
                          respiratory_rate=15)

    khodadad2018 = nk.rsp_clean(rsp, sampling_rate=1000, method="khodadad2018")
    assert len(rsp) == len(khodadad2018)

    biosppy = nk.rsp_clean(rsp, sampling_rate=1000, method="biosppy")
    assert len(rsp) == len(biosppy)


    # check if filter was applied
    fft_raw = np.fft.rfft(rsp)
    fft_khodadad2018 = np.fft.rfft(khodadad2018)
    fft_biosppy = np.fft.rfft(biosppy)

    freqs = np.fft.rfftfreq(len(rsp), 1/sampling_rate)
#    assert np.sum(fft_raw[freqs > 2]) > np.sum(fft_khodadad2018[freqs > 2])
    assert np.sum(fft_raw[freqs > 2]) > np.sum(fft_biosppy[freqs > 2])
    assert np.sum(fft_khodadad2018[freqs > 2]) > np.sum(fft_biosppy[freqs > 2])

    # check if detrending was applied
    assert np.mean(rsp) > np.mean(khodadad2018)


def test_rsp_findpeaks():

    rsp = nk.rsp_simulate(duration=120, sampling_rate=1000,
                          respiratory_rate=15)
    rsp_cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    signals, info = nk.rsp_findpeaks(rsp_cleaned)
    assert signals.shape == (120000, 2)
    assert signals["RSP_Peaks"].sum() == 28
    assert signals["RSP_Troughs"].sum() == 28
    assert info["RSP_Peaks"].shape[0] == 28
    assert info["RSP_Troughs"].shape[0] == 28
    # assert that extrema start with a trough and end with a peak
    assert info["RSP_Peaks"][0] > info["RSP_Troughs"][0]
    assert info["RSP_Peaks"][-1] > info["RSP_Troughs"][-1]


def test_rsp_rate():

    rsp = nk.rsp_simulate(duration=120, sampling_rate=1000,
                          respiratory_rate=15, method="sinusoidal")
    rsp_cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    signals, info = nk.rsp_findpeaks(rsp_cleaned)

    # vary desired_lenght over tests

    # test with peaks only
    test_length = 30
    data = nk.rsp_rate(peaks=info, sampling_rate=1000,
                       desired_length=test_length)
    assert data.shape == (test_length, 2)
    assert np.abs(data["RSP_Rate"].mean() - 15) < 0.2
    assert int(data["RSP_Amplitude"].mean()) == 2003

    # test with peaks and troughs passed in separately
    test_length = 300
    data = nk.rsp_rate(peaks=info["RSP_Peaks"], troughs=info["RSP_Troughs"],
                       sampling_rate=1000, desired_length=test_length)
    assert data.shape == (test_length, 2)
    assert np.abs(data["RSP_Rate"].mean() - 15) < 0.2
    assert int(data["RSP_Amplitude"].mean()) == 2003

    # test with DataFrame containing peaks and troughs
    data = nk.rsp_rate(signals, sampling_rate=1000)
    assert data.shape == (signals.shape[0], 2)
    assert np.abs(data["RSP_Rate"].mean() - 15) < 0.2
    assert int(data["RSP_Amplitude"].mean()) == 2003

    # test with dict containing peaks and troughs
    test_length = 30000
    data = nk.rsp_rate(info, sampling_rate=1000, desired_length=test_length)
    assert data.shape == (test_length, 2)
    assert np.abs(data["RSP_Rate"].mean() - 15) < 0.2
    assert int(data["RSP_Amplitude"].mean()) == 2003


def test_rsp_process():

    rsp = nk.rsp_simulate(duration=120, sampling_rate=1000,
                          respiratory_rate=15)
    signals, info = nk.rsp_process(rsp, sampling_rate=1000)

    # only check array dimensions since functions called by rsp_process have
    # already been unit tested
    assert signals.shape == (120000, 6)


def test_rsp_plot():

    rsp = nk.rsp_simulate(duration=120, sampling_rate=1000,
                          respiratory_rate=15)
    rsp_summary, _ = nk.rsp_process(rsp, sampling_rate=1000)
    nk.rsp_plot(rsp_summary)
    # this will identify the latest figure
    fig = plt.gcf()
    assert len(fig.axes) == 3
    titles = ["Signal and Breathing Extrema",
              "Breathing Rate",
              "Breathing Amplitude"]
    for (ax, title) in zip(fig.get_axes(), titles):
        assert ax.get_title() == title
    plt.close(fig)

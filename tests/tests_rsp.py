# -*- coding: utf-8 -*-
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

import biosppy

np.random.seed(42)

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

    rsp_biosppy = nk.rsp_clean(rsp, sampling_rate=1000, method="biosppy")
    assert len(rsp) == len(rsp_biosppy)


    # Check if filter was applied.
    fft_raw = np.fft.rfft(rsp)
    fft_khodadad2018 = np.fft.rfft(khodadad2018)
    fft_biosppy = np.fft.rfft(rsp_biosppy)

    freqs = np.fft.rfftfreq(len(rsp), 1/sampling_rate)
#    assert np.sum(fft_raw[freqs > 2]) > np.sum(fft_khodadad2018[freqs > 2])
    assert np.sum(fft_raw[freqs > 2]) > np.sum(fft_biosppy[freqs > 2])
    assert np.sum(fft_khodadad2018[freqs > 2]) > np.sum(fft_biosppy[freqs > 2])

    # Check if detrending was applied.
    assert np.mean(rsp) > np.mean(khodadad2018)

    # Comparison to biosppy (https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/resp.py#L62)
    rsp_biosppy = nk.rsp_clean(rsp, sampling_rate=sampling_rate, method="biosppy")
    original, _, _ = biosppy.tools.filter_signal(signal=rsp,
                                                 ftype='butter',
                                                 band='bandpass',
                                                 order=2,
                                                 frequency=[0.1, 0.35],
                                                 sampling_rate=sampling_rate)
    original = nk.signal_detrend(original, order=0)
    assert np.allclose((rsp_biosppy - original).mean(), 0, atol=1e-6)


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
    assert np.allclose(info["RSP_Peaks"].sum(), 1643787)
    assert np.allclose(info["RSP_Troughs"].sum(), 1586275)
    # Assert that extrema start with a trough and end with a peak.
    assert info["RSP_Peaks"][0] > info["RSP_Troughs"][0]
    assert info["RSP_Peaks"][-1] > info["RSP_Troughs"][-1]


def test_rsp_rate():

    rsp = nk.rsp_simulate(duration=120, sampling_rate=1000,
                          respiratory_rate=15, method="sinusoidal", noise=0)
    rsp_cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    signals, info = nk.rsp_findpeaks(rsp_cleaned)

    # Test with dictionary.
    test_length = 30
    rate = nk.rsp_rate(peaks=info, sampling_rate=1000,
                       desired_length=test_length)
    assert rate.shape == (test_length, 1)
    assert np.abs(rate["RSP_Rate"].mean() - 15) < 0.2

    # Test with DataFrame.
    rate = nk.rsp_rate(signals, sampling_rate=1000)
    assert rate.shape == (signals.shape[0], 1)
    assert np.abs(rate["RSP_Rate"].mean() - 15) < 0.2


def test_rsp_amplitude():

    rsp = nk.rsp_simulate(duration=120, sampling_rate=1000,
                          respiratory_rate=15, method="sinusoidal", noise=0)
    rsp_cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    signals, info = nk.rsp_findpeaks(rsp_cleaned)

    # Test with dictionary.
    test_length = 60
    amplitude = nk.rsp_amplitude(rsp_signal=rsp, extrema=info,
                                 desired_length=test_length)
    assert amplitude.shape == (test_length, 1)
    assert np.abs(amplitude["RSP_Amplitude"].mean() - 1) < 0.01

    # Test with DataFrame.
    amplitude = nk.rsp_amplitude(rsp_signal=rsp, extrema=signals)
    assert amplitude.shape == (rsp.size, 1)
    assert np.abs(amplitude["RSP_Amplitude"].mean() - 1) < 0.01


def test_rsp_process():

    rsp = nk.rsp_simulate(duration=120, sampling_rate=1000,
                          respiratory_rate=15)
    signals, info = nk.rsp_process(rsp, sampling_rate=1000)

    # Only check array dimensions since functions called by rsp_process have
    # already been unit tested.
    assert signals.shape == (120000, 6)
    assert np.array(["RSP_Raw",
                     "RSP_Clean",
                     "RSP_Peaks",
                     "RSP_Troughs",
                     "RSP_Rate",
                     "RSP_Amplitude"]) in signals.columns.values


def test_rsp_plot():

    rsp = nk.rsp_simulate(duration=120, sampling_rate=1000,
                          respiratory_rate=15)
    rsp_summary, _ = nk.rsp_process(rsp, sampling_rate=1000)
    nk.rsp_plot(rsp_summary)
    # This will identify the latest figure.
    fig = plt.gcf()
    assert len(fig.axes) == 3
    titles = ["Raw and Cleaned RSP",
              "Breathing Rate",
              "Breathing Amplitude"]
    for (ax, title) in zip(fig.get_axes(), titles):
        assert ax.get_title() == title
    plt.close(fig)

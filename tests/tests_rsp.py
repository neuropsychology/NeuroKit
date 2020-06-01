# -*- coding: utf-8 -*-
import biosppy
import matplotlib.pyplot as plt
import numpy as np

import neurokit2 as nk


def test_rsp_simulate():
    rsp1 = nk.rsp_simulate(duration=20, length=3000, random_state=42)
    assert len(rsp1) == 3000

    rsp2 = nk.rsp_simulate(duration=20, length=3000, respiratory_rate=80,
                           random_state=42)
#    pd.DataFrame({"RSP1":rsp1, "RSP2":rsp2}).plot()
#    pd.DataFrame({"RSP1":rsp1, "RSP2":rsp2}).hist()
    assert (len(nk.signal_findpeaks(rsp1, height_min=0.2)["Peaks"]) <
            len(nk.signal_findpeaks(rsp2, height_min=0.2)["Peaks"]))

    rsp3 = nk.rsp_simulate(duration=20, length=3000, method="sinusoidal",
                           random_state=42)
    rsp4 = nk.rsp_simulate(duration=20, length=3000, method="breathmetrics",
                           random_state=42)
#    pd.DataFrame({"RSP3":rsp3, "RSP4":rsp4}).plot()
    assert (len(nk.signal_findpeaks(rsp3, height_min=0.2)["Peaks"]) >
            len(nk.signal_findpeaks(rsp4, height_min=0.2)["Peaks"]))


def test_rsp_clean():

    sampling_rate = 100
    duration = 120
    rsp = nk.rsp_simulate(duration=duration, sampling_rate=sampling_rate,
                          respiratory_rate=15, noise=.1, random_state=42)
    # Add linear drift (to test baseline removal).
    rsp += nk.signal_distort(rsp, sampling_rate=sampling_rate,
                             linear_drift=True)

    khodadad2018 = nk.rsp_clean(rsp, sampling_rate=sampling_rate,
                                method="khodadad2018")
    assert len(rsp) == len(khodadad2018)

    rsp_biosppy = nk.rsp_clean(rsp, sampling_rate=sampling_rate,
                               method="biosppy")
    assert len(rsp) == len(rsp_biosppy)

    # Check if filter was applied.
    fft_raw = np.abs(np.fft.rfft(rsp))
    fft_khodadad2018 = np.abs(np.fft.rfft(khodadad2018))
    fft_biosppy = np.abs(np.fft.rfft(rsp_biosppy))

    freqs = np.fft.rfftfreq(len(rsp), 1 / sampling_rate)

    assert np.sum(fft_raw[freqs > 3]) > np.sum(fft_khodadad2018[freqs > 3])
    assert np.sum(fft_raw[freqs < .05]) > np.sum(fft_khodadad2018[freqs < .05])
    assert np.sum(fft_raw[freqs > .35]) > np.sum(fft_biosppy[freqs > .35])
    assert np.sum(fft_raw[freqs < .1]) > np.sum(fft_biosppy[freqs < .1])

    # Comparison to biosppy (https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/resp.py#L62)
    rsp_biosppy = nk.rsp_clean(rsp, sampling_rate=sampling_rate,
                               method="biosppy")
    original, _, _ = biosppy.tools.filter_signal(signal=rsp,
                                                 ftype='butter',
                                                 band='bandpass',
                                                 order=2,
                                                 frequency=[0.1, 0.35],
                                                 sampling_rate=sampling_rate)
    original = nk.signal_detrend(original, order=0)
    assert np.allclose((rsp_biosppy - original).mean(), 0, atol=1e-6)


def test_rsp_peaks():

    rsp = nk.rsp_simulate(duration=120, sampling_rate=1000,
                          respiratory_rate=15, random_state=42)
    rsp_cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    signals, info = nk.rsp_peaks(rsp_cleaned)
    assert signals.shape == (120000, 2)
    assert signals["RSP_Peaks"].sum() == 28
    assert signals["RSP_Troughs"].sum() == 28
    assert info["RSP_Peaks"].shape[0] == 28
    assert info["RSP_Troughs"].shape[0] == 28
    assert np.allclose(info["RSP_Peaks"].sum(), 1643817)
    assert np.allclose(info["RSP_Troughs"].sum(), 1586588)
    # Assert that extrema start with a trough and end with a peak.
    assert info["RSP_Peaks"][0] > info["RSP_Troughs"][0]
    assert info["RSP_Peaks"][-1] > info["RSP_Troughs"][-1]


def test_rsp_amplitude():

    rsp = nk.rsp_simulate(duration=120, sampling_rate=1000,
                          respiratory_rate=15, method="sinusoidal", noise=0)
    rsp_cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    signals, info = nk.rsp_peaks(rsp_cleaned)

    # Test with dictionary.
    amplitude = nk.rsp_amplitude(rsp, signals)
    assert amplitude.shape == (rsp.size, )
    assert np.abs(amplitude.mean() - 1) < 0.01

    # Test with DataFrame.
    amplitude = nk.rsp_amplitude(rsp, info)
    assert amplitude.shape == (rsp.size, )
    assert np.abs(amplitude.mean() - 1) < 0.01


def test_rsp_process():

    rsp = nk.rsp_simulate(duration=120, sampling_rate=1000,
                          respiratory_rate=15)
    signals, info = nk.rsp_process(rsp, sampling_rate=1000)

    # Only check array dimensions since functions called by rsp_process have
    # already been unit tested.
    assert signals.shape == (120000, 8)
    assert np.array(["RSP_Raw",
                     "RSP_Clean",
                     "RSP_Amplitude",
                     "RSP_Rate",
                     "RSP_Phase",
                     "RSP_PhaseCompletion",
                     "RSP_Peaks",
                     "RSP_Troughs",]) in signals.columns.values


def test_rsp_plot():

    rsp = nk.rsp_simulate(duration=120, sampling_rate=1000,
                          respiratory_rate=15)
    rsp_summary, _ = nk.rsp_process(rsp, sampling_rate=1000)
    nk.rsp_plot(rsp_summary)
    # This will identify the latest figure.
    fig = plt.gcf()
    assert len(fig.axes) == 3
    titles = ["Raw and Cleaned Signal",
              "Breathing Rate",
              "Breathing Amplitude"]
    for (ax, title) in zip(fig.get_axes(), titles):
        assert ax.get_title() == title
    plt.close(fig)


def test_rsp_eventrelated():

    rsp, info = nk.rsp_process(nk.rsp_simulate(duration=30, random_state=42))
    epochs = nk.epochs_create(rsp,
                              events=[5000, 10000, 15000],
                              epochs_start=-0.1,
                              epochs_end=1.9)
    rsp_eventrelated = nk.rsp_eventrelated(epochs)

    # Test rate features
    assert np.alltrue(np.array(rsp_eventrelated["RSP_Rate_Min"]) <
                      np.array(rsp_eventrelated["RSP_Rate_Mean"]))

    assert np.alltrue(np.array(rsp_eventrelated["RSP_Rate_Mean"]) <
                      np.array(rsp_eventrelated["RSP_Rate_Max"]))

    # Test amplitude features
    assert np.alltrue(np.array(rsp_eventrelated["RSP_Amplitude_Min"]) <
                      np.array(rsp_eventrelated["RSP_Amplitude_Mean"]))

    assert np.alltrue(np.array(rsp_eventrelated["RSP_Amplitude_Mean"]) <
                      np.array(rsp_eventrelated["RSP_Amplitude_Max"]))

    assert len(rsp_eventrelated["Label"]) == 3

def test_rsp_rrv():

    rsp90 = nk.rsp_simulate(duration=60, sampling_rate=1000, respiratory_rate=90, random_state=42)
    rsp110 = nk.rsp_simulate(duration=60, sampling_rate=1000, respiratory_rate=110, random_state=42)

    cleaned90 = nk.rsp_clean(rsp90, sampling_rate=1000)
    _, peaks90 = nk.rsp_peaks(cleaned90)
    rsp_rate90 = nk.signal_rate(peaks90, desired_length=len(rsp90))

    cleaned110 = nk.rsp_clean(rsp110, sampling_rate=1000)
    _, peaks110 = nk.rsp_peaks(cleaned110)
    rsp_rate110 = nk.signal_rate(peaks110, desired_length=len(rsp110))

    rsp90_rrv = nk.rsp_rrv(rsp_rate90, peaks90)
    rsp110_rrv = nk.rsp_rrv(rsp_rate110, peaks110)

    assert np.array(rsp90_rrv["RRV_SDBB"]) < np.array(rsp110_rrv["RRV_SDBB"])
    assert np.array(rsp90_rrv["RRV_RMSSD"]) < np.array(rsp110_rrv["RRV_RMSSD"])
    assert np.array(rsp90_rrv["RRV_SDSD"]) < np.array(rsp110_rrv["RRV_SDSD"])
    # assert np.array(rsp90_rrv["RRV_pNN50"]) == np.array(rsp110_rrv["RRV_pNN50"]) == np.array(rsp110_rrv["RRV_pNN20"]) == np.array(rsp90_rrv["RRV_pNN20"]) == 0
    # assert np.array(rsp90_rrv["RRV_TINN"]) < np.array(rsp110_rrv["RRV_TINN"])
    # assert np.array(rsp90_rrv["RRV_HTI"]) > np.array(rsp110_rrv["RRV_HTI"])
    assert np.array(rsp90_rrv["RRV_HF"]) < np.array(rsp110_rrv["RRV_HF"])
    assert np.isnan(rsp90_rrv["RRV_LF"][0])
    assert np.isnan(rsp110_rrv["RRV_LF"][0])

#    assert all(elem in ['RRV_SDBB','RRV_RMSSD', 'RRV_SDSD'
#                        'RRV_VLF', 'RRV_LF', 'RRV_HF', 'RRV_LFHF',
#                        'RRV_LFn', 'RRV_HFn',
#                        'RRV_SD1', 'RRV_SD2', 'RRV_SD2SD1','RRV_ApEn', 'RRV_SampEn', 'RRV_DFA']
#               for elem in np.array(rsp110_rrv.columns.values, dtype=str))


def test_rsp_intervalrelated():

    data = nk.data("bio_resting_5min_100hz")
    df, info = nk.rsp_process(data["RSP"], sampling_rate=100)

    # Test with signal dataframe
    features_df = nk.rsp_intervalrelated(df)

    assert features_df.shape[0] == 1  # Number of rows

    # Test with dict
    epochs = nk.epochs_create(df, events=[0, 15000],
                              sampling_rate=100, epochs_end=150)
    features_dict = nk.rsp_intervalrelated(epochs)

    assert features_dict.shape[0] == 2  # Number of rows

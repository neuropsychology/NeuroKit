# -*- coding: utf-8 -*-
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

import biosppy

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
    ecg_biosppy = nk.ecg_clean(ecg, sampling_rate=sampling_rate, method="biosppy")
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

    # Test without request to return artifacts.
    signals, info = nk.ecg_peaks(ecg_cleaned_nk, method="neurokit")

    assert signals.shape == (120000, 1)
    assert np.allclose(signals["ECG_R_Peaks"].values.sum(dtype=np.int64), 152, atol=1)
#    assert np.allclose(info["ECG_R_Peaks"].sum(dtype=np.int64), 9283853, atol=1)

    # Test with request to return artifacts.
    signals, info, artifacts = nk.ecg_peaks(ecg_cleaned_nk,
                                            return_artifacts=True,
                                            method="neurokit")

    assert signals.shape == (120000, 1)
    assert np.allclose(signals["ECG_R_Peaks"].values.sum(dtype=np.int64), 152, atol=1)
#    assert np.allclose(info["ECG_R_Peaks"].sum(dtype=np.int64), 9283853, atol=1)
    assert all(isinstance(x, int) for x in artifacts["ectopic"])
    assert all(isinstance(x, int) for x in artifacts["missed"])
    assert all(isinstance(x, int) for x in artifacts["extra"])
    assert all(isinstance(x, int) for x in artifacts["longshort"])

def test_ecg_rate():

    sampling_rate = 1000
    noise = 0.15

    ecg = nk.ecg_simulate(duration=120, sampling_rate=sampling_rate,
                          noise=noise, random_state=42)
    ecg_cleaned_nk = nk.ecg_clean(ecg, sampling_rate=sampling_rate,
                                  method="neurokit")

    signals, info, artifacts = nk.ecg_peaks(ecg_cleaned_nk,
                                            return_artifacts=True,
                                            method="neurokit")

    # Test without artifact correction and without desired length.
    rate = nk.ecg_rate(rpeaks=info, sampling_rate=sampling_rate)

    assert rate.shape == (info["ECG_R_Peaks"].size, )
    assert np.allclose(rate.mean(), 81, atol=2)

    # Test without artifact correction and with desired length.
    test_length = 1200
    rate = nk.ecg_rate(rpeaks=info, sampling_rate=sampling_rate,
                       desired_length=test_length)

    assert rate.shape == (test_length, )
    assert np.allclose(rate.mean(), 81, atol=2)

    # Test with artifact correction and without desired length.
    rate = nk.ecg_rate(rpeaks=info, artifacts=artifacts,
                       sampling_rate=sampling_rate)

    assert rate.shape == (143, )
    assert np.allclose(rate.mean(), 75, atol=1)

    # Test with artifact correction and with desired length.
    test_length = 1200
    rate = nk.ecg_rate(rpeaks=info, sampling_rate=sampling_rate,
                       artifacts=artifacts, desired_length=test_length)

    assert rate.shape == (test_length, )
    assert np.allclose(rate.mean(), 75, atol=1)


def test_ecg_fixpeaks():

    sampling_rate = 1000
    noise = 0.15

    ecg = nk.ecg_simulate(duration=120, sampling_rate=sampling_rate,
                          noise=noise, random_state=42)

    rpeaks = nk.ecg_findpeaks(ecg)

    artifacts = nk.ecg_fixpeaks(rpeaks)

    assert all(isinstance(x, int) for x in artifacts["ectopic"])
    assert all(isinstance(x, int) for x in artifacts["missed"])
    assert all(isinstance(x, int) for x in artifacts["extra"])
    assert all(isinstance(x, int) for x in artifacts["longshort"])

    # TODO: simulate speific types of artifacts at specific indices and assert
    # their detection.


def test_ecg_process():

    sampling_rate = 1000
    noise = 0.05

    ecg = nk.ecg_simulate(sampling_rate=sampling_rate, noise=noise)
    signals, info = nk.ecg_process(ecg, sampling_rate=sampling_rate,
                                   method="neurokit")
    # Only check array dimensions and column names since functions called by
    # ecg_process have already been unit tested
    assert all(elem in ["ECG_Raw", "ECG_Clean", "ECG_R_Peaks", "ECG_Rate"]
               for elem in np.array(signals.columns.values, dtype=str))


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
    plt.close(fig)

    # Plot data over seconds.
    nk.ecg_plot(ecg_summary, sampling_rate=1000)
    # This will identify the latest figure.
    fig = plt.gcf()
    assert len(fig.axes) == 2
    titles = ["Raw and Cleaned Signal",
              "Heart Rate"]
    for (ax, title) in zip(fig.get_axes(), titles):
        assert ax.get_title() == title
    assert fig.get_axes()[1].get_xlabel() == "Time (seconds)"
    plt.close(fig)

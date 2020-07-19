import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.signal

import neurokit2 as nk

# =============================================================================
# Signal
# =============================================================================


def test_signal_simulate():
    # Warning for nyquist criterion
    with pytest.warns(
        nk.misc.NeuroKitWarning,
        match=r"Skipping requested frequency.*cannot be resolved.*"
    ):
        nk.signal_simulate(sampling_rate=100, frequency=11, silent=False)

    # Warning for period duration
    with pytest.warns(
        nk.misc.NeuroKitWarning,
        match=r"Skipping requested frequency.*since its period of.*"
    ):
        nk.signal_simulate(duration=1, frequency=0.1, silent=False)


def test_signal_smooth():

    # TODO: test kernels other than "boxcar"
    signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    smooth1 = nk.signal_smooth(signal, kernel="boxcar", size=100)
    smooth2 = nk.signal_smooth(signal, kernel="boxcar", size=500)
    # assert that the signal's amplitude is attenuated more with wider kernels
    assert np.allclose(np.std(smooth1), 0.6044, atol=0.00001)
    assert np.allclose(np.std(smooth2), 0.1771, atol=0.0001)


def test_signal_binarize():

    signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    binary = nk.signal_binarize(signal)
    assert len(binary) == 1000

    binary = nk.signal_binarize(list(signal))
    assert len(binary) == 1000


def test_signal_resample():

    signal = np.cos(np.linspace(start=0, stop=20, num=50))

    downsampled_interpolation = nk.signal_resample(
        signal, method="interpolation", sampling_rate=1000, desired_sampling_rate=500
    )
    downsampled_numpy = nk.signal_resample(signal, method="numpy", sampling_rate=1000, desired_sampling_rate=500)
    downsampled_pandas = nk.signal_resample(signal, method="pandas", sampling_rate=1000, desired_sampling_rate=500)
    downsampled_fft = nk.signal_resample(signal, method="FFT", sampling_rate=1000, desired_sampling_rate=500)
    downsampled_poly = nk.signal_resample(signal, method="poly", sampling_rate=1000, desired_sampling_rate=500)

    # Upsample
    upsampled_interpolation = nk.signal_resample(
        downsampled_interpolation, method="interpolation", sampling_rate=500, desired_sampling_rate=1000
    )
    upsampled_numpy = nk.signal_resample(
        downsampled_numpy, method="numpy", sampling_rate=500, desired_sampling_rate=1000
    )
    upsampled_pandas = nk.signal_resample(
        downsampled_pandas, method="pandas", sampling_rate=500, desired_sampling_rate=1000
    )
    upsampled_fft = nk.signal_resample(downsampled_fft, method="FFT", sampling_rate=500, desired_sampling_rate=1000)
    upsampled_poly = nk.signal_resample(downsampled_poly, method="poly", sampling_rate=500, desired_sampling_rate=1000)

    # Check
    rez = pd.DataFrame(
        {
            "Interpolation": upsampled_interpolation - signal,
            "Numpy": upsampled_numpy - signal,
            "Pandas": upsampled_pandas - signal,
            "FFT": upsampled_fft - signal,
            "Poly": upsampled_poly - signal,
        }
    )
    assert np.allclose(np.mean(rez.mean()), 0.0001, atol=0.0001)


def test_signal_detrend():

    signal = np.cos(np.linspace(start=0, stop=10, num=1000))  # Low freq
    signal += np.cos(np.linspace(start=0, stop=100, num=1000))  # High freq
    signal += 3  # Add baseline

    rez_nk = nk.signal_detrend(signal, order=1)
    rez_scipy = scipy.signal.detrend(signal, type="linear")
    assert np.allclose(np.mean(rez_nk - rez_scipy), 0, atol=0.000001)

    rez_nk = nk.signal_detrend(signal, order=0)
    rez_scipy = scipy.signal.detrend(signal, type="constant")
    assert np.allclose(np.mean(rez_nk - rez_scipy), 0, atol=0.000001)

    # Tarvainen
    rez_nk = nk.signal_detrend(signal, method="tarvainen2002", regularization=500)
    assert np.allclose(np.mean(rez_nk - signal), -2.88438737697, atol=0.000001)


def test_signal_filter():

    signal = np.cos(np.linspace(start=0, stop=10, num=1000))  # Low freq
    signal += np.cos(np.linspace(start=0, stop=100, num=1000))  # High freq
    filtered = nk.signal_filter(signal, highcut=10)
    assert np.std(signal) > np.std(filtered)

    with pytest.warns(nk.misc.NeuroKitWarning, match=r"The sampling rate is too low.*"):
        with pytest.raises(ValueError):
            nk.signal_filter(signal, method="bessel", sampling_rate=100 ,highcut=50)

    # Generate 10 seconds of signal with 2 Hz oscillation and added 50Hz powerline-noise.
    sampling_rate = 250
    samples = np.arange(10 * sampling_rate)

    signal = np.sin(2 * np.pi * 2 * (samples / sampling_rate))
    powerline = np.sin(2 * np.pi * 50 * (samples / sampling_rate))

    signal_corrupted = signal + powerline
    signal_clean = nk.signal_filter(signal_corrupted, sampling_rate=sampling_rate, method="powerline")

    # import matplotlib.pyplot as plt
    # figure, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
    # ax0.plot(signal_corrupted)
    # ax1.plot(signal)
    # ax2.plot(signal_clean * 100)

    assert np.allclose(sum(signal_clean - signal), -2, atol=0.2)


def test_signal_interpolate():

    x_axis = np.linspace(start=10, stop=30, num=10)
    signal = np.cos(x_axis)

    interpolated = nk.signal_interpolate(x_axis, signal, x_new=np.arange(1000))
    assert len(interpolated) == 1000
    assert interpolated[0] == signal[0]
    assert interpolated[-1] == signal[-1]


def test_signal_findpeaks():

    signal1 = np.cos(np.linspace(start=0, stop=30, num=1000))
    info1 = nk.signal_findpeaks(signal1)

    signal2 = np.concatenate([np.arange(0, 20, 0.1), np.arange(17, 30, 0.1), np.arange(30, 10, -0.1)])
    info2 = nk.signal_findpeaks(signal2)
    assert len(info1["Peaks"]) > len(info2["Peaks"])


def test_signal_merge():

    signal1 = np.cos(np.linspace(start=0, stop=10, num=100))
    signal2 = np.cos(np.linspace(start=0, stop=20, num=100))

    signal = nk.signal_merge(signal1, signal2, time1=[0, 10], time2=[-5, 5])
    assert len(signal) == 150
    assert signal[0] == signal2[0] + signal2[0]


def test_signal_rate():  # since singal_rate wraps signal_period, the latter is tested as well

    # Test with array.
    duration = 10
    sampling_rate = 1000
    signal = nk.signal_simulate(duration=duration, sampling_rate=sampling_rate, frequency=1)
    info = nk.signal_findpeaks(signal)
    rate = nk.signal_rate(peaks=info["Peaks"], sampling_rate=1000, desired_length=len(signal))
    assert rate.shape[0] == duration * sampling_rate

    # Test with dictionary.produced from signal_findpeaks.
    assert info[list(info.keys())[0]].shape == (info["Peaks"].shape[0],)

    # Test with DataFrame.
    duration = 120
    sampling_rate = 1000
    rsp = nk.rsp_simulate(
        duration=duration, sampling_rate=sampling_rate, respiratory_rate=15, method="sinuosoidal", noise=0
    )
    rsp_cleaned = nk.rsp_clean(rsp, sampling_rate=sampling_rate)
    signals, info = nk.rsp_peaks(rsp_cleaned)
    rate = nk.signal_rate(signals, sampling_rate=sampling_rate, desired_length=duration * sampling_rate)
    assert rate.shape == (signals.shape[0],)

    # Test with dictionary.produced from rsp_findpeaks.
    rate = nk.signal_rate(info, sampling_rate=sampling_rate, desired_length=duration * sampling_rate)
    assert rate.shape == (duration * sampling_rate,)


def test_signal_period():
    # Test warning path of no peaks
    with pytest.warns(nk.NeuroKitWarning, match=r"Too few peaks detected to compute the rate."):
        nk.signal_period(np.zeros)


def test_signal_plot():

    # Test with array
    signal = nk.signal_simulate(duration=10, sampling_rate=1000)
    nk.signal_plot(signal, sampling_rate=1000)
    fig = plt.gcf()
    for ax in fig.get_axes():
        handles, labels = ax.get_legend_handles_labels()
    assert labels == ["Signal"]
    assert len(labels) == len(handles) == len([signal])
    assert ax.get_xlabel() == "Time (seconds)"
    plt.close(fig)

    # Test with dataframe
    data = pd.DataFrame(
        {
            "Signal2": np.cos(np.linspace(start=0, stop=20, num=1000)),
            "Signal3": np.sin(np.linspace(start=0, stop=20, num=1000)),
            "Signal4": nk.signal_binarize(np.cos(np.linspace(start=0, stop=40, num=1000))),
        }
    )
    nk.signal_plot(data, sampling_rate=None)
    fig = plt.gcf()
    for ax in fig.get_axes():
        handles, labels = ax.get_legend_handles_labels()
    assert labels == list(data.columns.values)
    assert len(labels) == len(handles) == len(data.columns)
    assert ax.get_xlabel() == "Samples"
    plt.close(fig)

    # Test with list
    signal = nk.signal_binarize(nk.signal_simulate(duration=10))
    phase = nk.signal_phase(signal, method="percents")
    nk.signal_plot([signal, phase])
    fig = plt.gcf()
    for ax in fig.get_axes():
        handles, labels = ax.get_legend_handles_labels()
    assert labels == ["Signal1", "Signal2"]
    assert len(labels) == len(handles) == len([signal, phase])
    assert ax.get_xlabel() == "Samples"
    plt.close(fig)


def test_signal_power():

    signal1 = nk.signal_simulate(duration=20, frequency=1, sampling_rate=500)
    pwr1 = nk.signal_power(signal1, [[0.9, 1.6], [1.4, 2.0]], sampling_rate=500)

    signal2 = nk.signal_simulate(duration=20, frequency=1, sampling_rate=100)
    pwr2 = nk.signal_power(signal2, [[0.9, 1.6], [1.4, 2.0]], sampling_rate=100)

    assert np.allclose(np.mean(pwr1.iloc[0] - pwr2.iloc[0]), 0, atol=0.01)


def test_signal_timefrequency():

    signal = nk.signal_simulate(duration = 50, frequency=5) + 2 * nk.signal_simulate(duration = 50, frequency=20)

    # short-time fourier transform
    frequency, time, stft = nk.signal_timefrequency(signal, method="stft", min_frequency=1, max_frequency=50, show=False)

    assert len(frequency) == stft.shape[0]
    assert len(time) == stft.shape[1]
    indices_freq5 = np.logical_and(frequency > 3, frequency < 7)
    indices_freq20 = np.logical_and(frequency > 18, frequency < 22)
    assert np.sum(stft[indices_freq5]) < np.sum(stft[indices_freq20])

    # wavelet transform
    frequency, time, cwtm = nk.signal_timefrequency(signal, method="cwt", max_frequency=50, show=False)

    assert len(frequency) == cwtm.shape[0]
    assert len(time) == cwtm.shape[1]
    indices_freq5 = np.logical_and(frequency > 3, frequency < 7)
    indices_freq20 = np.logical_and(frequency > 18, frequency < 22)
    assert np.sum(cwtm[indices_freq5]) < np.sum(cwtm[indices_freq20])

    # wvd
    frequency, time, wvd = nk.signal_timefrequency(signal, method="wvd", max_frequency=50, show=False)
    assert len(frequency) == wvd.shape[0]
    assert len(time) == wvd.shape[1]
    indices_freq5 = np.logical_and(frequency > 3, frequency < 7)
    indices_freq20 = np.logical_and(frequency > 18, frequency < 22)
    assert np.sum(wvd[indices_freq5]) < np.sum(wvd[indices_freq20])

    # pwvd
    frequency, time, pwvd = nk.signal_timefrequency(signal, method="pwvd",
                                                    max_frequency=50, show=False)
    assert len(frequency) == pwvd.shape[0]
    assert len(time) == pwvd.shape[1]
    indices_freq5 = np.logical_and(frequency > 3, frequency < 7)
    indices_freq20 = np.logical_and(frequency > 18, frequency < 22)
    assert np.sum(pwvd[indices_freq5]) < np.sum(pwvd[indices_freq20])

def test_signal_psd(recwarn):
    warnings.simplefilter("always")

    data = nk.data("bio_eventrelated_100hz")
    out = nk.signal_psd(data["ECG"], sampling_rate=100)

    assert list(out.columns) == ["Frequency", "Power"]

    assert len(recwarn) == 1
    assert recwarn.pop(nk.misc.NeuroKitWarning)


def test_signal_distort():
    signal = nk.signal_simulate(duration=10, frequency=0.5, sampling_rate=10)

    # Warning for nyquist criterion
    with pytest.warns(
        nk.misc.NeuroKitWarning,
        match=r"Skipping requested noise frequency.*cannot be resolved.*"
    ):
        nk.signal_distort(signal, sampling_rate=10, noise_amplitude=1, silent=False)

    # Warning for period duration
    with pytest.warns(
        nk.misc.NeuroKitWarning,
        match=r"Skipping requested noise frequency.*since its period of.*"
    ):
        signal = nk.signal_simulate(duration=1, frequency=1, sampling_rate=10)
        nk.signal_distort(signal, noise_amplitude=1, noise_frequency=0.1, silent=False)


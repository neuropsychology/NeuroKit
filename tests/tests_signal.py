import numpy as np
import pandas as pd
import neurokit2 as nk
import scipy.signal
import biosppy


# =============================================================================
# Signal
# =============================================================================

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

    downsampled_interpolation = nk.signal_resample(signal,
                                                   method="interpolation",
                                                   sampling_rate=1000,
                                                   desired_sampling_rate=500)
    downsampled_numpy = nk.signal_resample(signal, method="numpy",
                                           sampling_rate=1000,
                                           desired_sampling_rate=500)
    downsampled_pandas = nk.signal_resample(signal, method="pandas",
                                            sampling_rate=1000,
                                            desired_sampling_rate=500)
    downsampled_fft = nk.signal_resample(signal, method="FFT",
                                         sampling_rate=1000,
                                         desired_sampling_rate=500)
    downsampled_poly = nk.signal_resample(signal, method="poly",
                                          sampling_rate=1000,
                                          desired_sampling_rate=500)

    # Upsample
    upsampled_interpolation = nk.signal_resample(downsampled_interpolation,
                                                 method="interpolation",
                                                 sampling_rate=500,
                                                 desired_sampling_rate=1000)
    upsampled_numpy = nk.signal_resample(downsampled_numpy, method="numpy",
                                         sampling_rate=500,
                                         desired_sampling_rate=1000)
    upsampled_pandas = nk.signal_resample(downsampled_pandas, method="pandas",
                                          sampling_rate=500,
                                          desired_sampling_rate=1000)
    upsampled_fft = nk.signal_resample(downsampled_fft, method="FFT",
                                       sampling_rate=500,
                                       desired_sampling_rate=1000)
    upsampled_poly = nk.signal_resample(downsampled_poly, method="poly",
                                        sampling_rate=500,
                                        desired_sampling_rate=1000)

    # Check
    rez = pd.DataFrame({"Interpolation": upsampled_interpolation - signal,
                        "Numpy": upsampled_numpy - signal,
                        "Pandas": upsampled_pandas - signal,
                        "FFT": upsampled_fft - signal,
                        "Poly": upsampled_poly - signal})
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

    signal = np.cos(np.linspace(start=0, stop=10, num=1000)) # Low freq
    signal += np.cos(np.linspace(start=0, stop=100, num=1000)) # High freq
    filtered = nk.signal_filter(signal, highcut=10)
    assert np.std(signal) > np.std(filtered)

def test_signal_interpolate():

    x_axis = np.linspace(start=10, stop=30, num=10)
    signal = np.cos(x_axis)

    interpolated = nk.signal_interpolate(signal, desired_length=1000)
    assert len(interpolated) == 1000

    new_x = np.linspace(start=0, stop=40, num=1000)
    interpolated = nk.signal_interpolate(signal,
                                         desired_length=1000,
                                         x_axis=x_axis,
                                         new_x=new_x)
    assert len(interpolated) == 1000
    assert interpolated[0] == signal[0]


def test_signal_findpeaks():

    signal1 = np.cos(np.linspace(start=0, stop=30, num=1000))
    info1 = nk.signal_findpeaks(signal1)

    signal2 = np.concatenate([np.arange(0, 20, 0.1), np.arange(17, 30, 0.1),
                              np.arange(30, 10, -0.1)])
    info2 = nk.signal_findpeaks(signal2)
    assert len(info1["Peaks"]) > len(info2["Peaks"])


def test_signal_merge():

    signal1 = np.cos(np.linspace(start=0, stop=10, num=100))
    signal2 = np.cos(np.linspace(start=0, stop=20, num=100))

    signal = nk.signal_merge(signal1, signal2, time1=[0, 10], time2=[-5, 5])
    assert len(signal) == 150
    assert signal[0] == signal2[0] + signal2[0]

def test_signal_rate():

    # Test with array.
    signal = nk.signal_simulate(duration=10, sampling_rate=1000,
                             frequency=1)
    info = nk.signal_findpeaks(signal)
    rate = nk.signal_rate(peaks=info["Peaks"], sampling_rate=1000,
                       desired_length=None)
    assert rate.shape[0] == np.max(info["Peaks"])

    # Test with dictionary.produced from signal_findpeaks.
    assert info[list(info.keys())[0]].shape == (info["Peaks"].shape[0], )

    # Test with DataFrame.
    rsp = nk.rsp_simulate(duration=120, sampling_rate=1000,
                       respiratory_rate=15, method="sinuosoidal", noise=0)
    rsp_cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    info, signals = nk.rsp_peaks(rsp_cleaned)
    rate = nk.signal_rate(signals, sampling_rate=1000)
    assert rate.shape == (signals.shape[0], )

    # Test with dictionary.produced from rsp_findpeaks.
    test_length = 30
    rate = nk.signal_rate(info, sampling_rate=1000,
                       desired_length=test_length)
    assert rate.shape == (test_length, )

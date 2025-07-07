# -*- coding: utf-8 -*-

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytest

import neurokit2 as nk

durations = (20, 200, 300)
sampling_rates = (25, 50, 500)
heart_rates = (50, 120)
freq_modulations = (0.1, 0.4)

params = [durations, sampling_rates, heart_rates, freq_modulations]

params_combis = list(itertools.product(*params))


@pytest.mark.parametrize(
    "duration, sampling_rate, heart_rate, freq_modulation", params_combis
)
def test_ppg_simulate(duration, sampling_rate, heart_rate, freq_modulation):
    ppg = nk.ppg_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        heart_rate=heart_rate,
        frequency_modulation=freq_modulation,
        ibi_randomness=0,
        drift=0,
        motion_amplitude=0,
        powerline_amplitude=0,
        burst_amplitude=0,
        burst_number=0,
        random_state=42,
        random_state_distort=42,
        show=False,
    )

    assert ppg.size == duration * sampling_rate

    signals, _ = nk.ppg_process(ppg, sampling_rate=sampling_rate)
    if sampling_rate > 25:
        assert np.allclose(signals["PPG_Rate"].mean(), heart_rate, atol=1)
        # Ensure that the heart rate fluctuates in the requested range.
        groundtruth_range = freq_modulation * heart_rate
        observed_range = np.percentile(signals["PPG_Rate"], 90) - np.percentile(
            signals["PPG_Rate"], 10
        )
        assert np.allclose(
            groundtruth_range, observed_range, atol=groundtruth_range * 0.20
        )

    # TODO: test influence of different noise configurations


@pytest.mark.parametrize(
    "ibi_randomness, std_heart_rate",
    [(0.1, 3), (0.2, 5), (0.3, 8), (0.4, 11), (0.5, 14), (0.6, 19)],
)
def test_ppg_simulate_ibi(ibi_randomness, std_heart_rate):
    ppg = nk.ppg_simulate(
        duration=20,
        sampling_rate=50,
        heart_rate=70,
        frequency_modulation=0,
        ibi_randomness=ibi_randomness,
        drift=0,
        motion_amplitude=0,
        powerline_amplitude=0,
        burst_amplitude=0,
        burst_number=0,
        random_state=42,
        show=False,
    )

    assert ppg.size == 20 * 50

    signals, _ = nk.ppg_process(ppg, sampling_rate=50)
    assert np.allclose(signals["PPG_Rate"].mean(), 70, atol=1.5)

    # Ensure that standard deviation of heart rate
    assert np.allclose(signals["PPG_Rate"].std(), std_heart_rate, atol=1)

    # TODO: test influence of different noise configurations


def test_ppg_simulate_legacy_rng():
    ppg = nk.ppg_simulate(
        duration=30,
        sampling_rate=250,
        heart_rate=70,
        frequency_modulation=0.2,
        ibi_randomness=0.1,
        drift=0.1,
        motion_amplitude=0.1,
        powerline_amplitude=0.01,
        random_state=654,
        random_state_distort="legacy",
        show=False,
    )

    # Run simple checks to verify that the signal is the same as that generated with version 0.2.3
    # before the introduction of the new random number generation approach
    assert np.allclose(np.mean(ppg), 0.6598246992405254)
    assert np.allclose(np.std(ppg), 0.4542274696384863)
    assert np.allclose(
        np.mean(np.reshape(ppg, (-1, 1500)), axis=1),
        [0.630608661400, 0.63061887029, 0.60807993168, 0.65731025466, 0.77250577818],
    )


def test_ppg_clean():
    sampling_rate = 500

    ppg = nk.ppg_simulate(
        duration=30,
        sampling_rate=sampling_rate,
        heart_rate=180,
        frequency_modulation=0.01,
        ibi_randomness=0.1,
        drift=1,
        motion_amplitude=0.5,
        powerline_amplitude=0.1,
        burst_amplitude=1,
        burst_number=5,
        random_state=42,
        show=False,
    )
    ppg_cleaned_elgendi = nk.ppg_clean(
        ppg, sampling_rate=sampling_rate, method="elgendi"
    )

    assert ppg.size == ppg_cleaned_elgendi.size

    # Assert that bandpass filter with .5 Hz lowcut and 8 Hz highcut was applied.
    fft_raw = np.abs(np.fft.rfft(ppg))
    fft_elgendi = np.abs(np.fft.rfft(ppg_cleaned_elgendi))

    freqs = np.fft.rfftfreq(ppg.size, 1 / sampling_rate)

    assert np.sum(fft_raw[freqs < 0.5]) > np.sum(fft_elgendi[freqs < 0.5])
    assert np.sum(fft_raw[freqs > 8]) > np.sum(fft_elgendi[freqs > 8])


def test_ppg_findpeaks():
    sampling_rate = 500

    # Test Elgendi method
    ppg = nk.ppg_simulate(
        duration=30,
        sampling_rate=sampling_rate,
        heart_rate=60,
        frequency_modulation=0.01,
        ibi_randomness=0.1,
        drift=1,
        motion_amplitude=0.5,
        powerline_amplitude=0.1,
        burst_amplitude=1,
        burst_number=5,
        random_state=42,
        show=True,
    )
    ppg_cleaned_elgendi = nk.ppg_clean(
        ppg, sampling_rate=sampling_rate, method="elgendi"
    )

    info_elgendi = nk.ppg_findpeaks(
        ppg_cleaned_elgendi, sampling_rate=sampling_rate, show=True
    )

    peaks = info_elgendi["PPG_Peaks"]

    assert peaks.size == 29
    assert np.abs(peaks.sum() - 219764) < 5  # off by no more than 5 samples in total

    # Test MSPTD method
    info_msptd = nk.ppg_findpeaks(
        ppg, sampling_rate=sampling_rate, method="bishop", show=True
    )

    peaks = info_msptd["PPG_Peaks"]

    assert peaks.size == 29
    assert np.abs(peaks.sum() - 219665) < 30  # off by no more than 30 samples in total


@pytest.mark.parametrize(
    "method_cleaning, method_peaks",
    [("elgendi", "elgendi"), ("nabian2018", "elgendi"), ("elgendi", "bishop")],
)
def test_ppg_report(tmp_path, method_cleaning, method_peaks):
    sampling_rate = 100

    ppg = nk.ppg_simulate(
        duration=30,
        sampling_rate=sampling_rate,
        heart_rate=60,
        frequency_modulation=0.01,
        ibi_randomness=0.1,
        drift=1,
        motion_amplitude=0.5,
        powerline_amplitude=0.1,
        burst_amplitude=1,
        burst_number=5,
        random_state=42,
        show=True,
    )

    d = tmp_path / "sub"
    d.mkdir()
    p = d / "myreport.html"

    signals, _ = nk.ppg_process(
        ppg,
        sampling_rate=sampling_rate,
        report=str(p),
        method_cleaning=method_cleaning,
        method_peaks=method_peaks,
    )
    assert p.is_file()


def test_ppg_intervalrelated():
    sampling_rate = 100

    ppg = nk.ppg_simulate(
        duration=500,
        sampling_rate=sampling_rate,
        heart_rate=70,
        frequency_modulation=0.025,
        ibi_randomness=0.15,
        drift=0.5,
        motion_amplitude=0.25,
        powerline_amplitude=0.25,
        burst_amplitude=0.5,
        burst_number=3,
        random_state=0,
        show=True,
    )
    # Process the data
    df, info = nk.ppg_process(ppg, sampling_rate=sampling_rate)
    epochs = nk.epochs_create(
        df, events=[0, 15000], sampling_rate=sampling_rate, epochs_end=150
    )
    epochs_ppg_intervals = nk.ppg_intervalrelated(epochs)
    assert "PPG_Rate_Mean" in epochs_ppg_intervals.columns

    ppg_intervals = nk.ppg_intervalrelated(df)
    assert "PPG_Rate_Mean" in ppg_intervals.columns


def test_ppg_plot():
    ppg = nk.ppg_simulate(duration=60, sampling_rate=250)

    ppg_summary, info = nk.ppg_process(ppg, sampling_rate=250)

    # Plot data over seconds.
    nk.ppg_plot(ppg_summary, info)
    fig = plt.gcf()
    assert len(fig.axes) == 3
    assert fig.get_axes()[1].get_xlabel() == "Time (seconds)"
    np.testing.assert_array_equal(fig.axes[0].get_xticks(), fig.axes[1].get_xticks())
    plt.close(fig)

    # Make sure it works with cropped data
    nk.ppg_plot(ppg_summary[0:1000], info)
    fig = plt.gcf()
    assert fig.get_axes()[2].get_xlabel() == "Time (seconds)"

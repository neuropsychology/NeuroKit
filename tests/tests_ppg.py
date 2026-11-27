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


@pytest.mark.parametrize("duration, sampling_rate, heart_rate, freq_modulation", params_combis)
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
        observed_range = np.percentile(signals["PPG_Rate"], 90) - np.percentile(signals["PPG_Rate"], 10)
        assert np.allclose(groundtruth_range, observed_range, atol=groundtruth_range * 0.20)

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
    ppg_cleaned_elgendi = nk.ppg_clean(ppg, sampling_rate=sampling_rate, method="elgendi")

    assert ppg.size == ppg_cleaned_elgendi.size

    # Assert that bandpass filter with .5 Hz lowcut and 8 Hz highcut was applied.
    fft_raw = np.abs(np.fft.rfft(ppg))
    fft_elgendi = np.abs(np.fft.rfft(ppg_cleaned_elgendi))

    freqs = np.fft.rfftfreq(ppg.size, 1 / sampling_rate)

    assert np.sum(fft_raw[freqs < 0.5]) > np.sum(fft_elgendi[freqs < 0.5])
    assert np.sum(fft_raw[freqs > 8]) > np.sum(fft_elgendi[freqs > 8])

    # Test langevin2021 method: bandpass 0.7–3.5 Hz, Butterworth order 2
    ppg_cleaned_langevin = nk.ppg_clean(ppg, sampling_rate=sampling_rate, method="langevin2021")
    assert ppg.size == ppg_cleaned_langevin.size

    fft_langevin = np.abs(np.fft.rfft(ppg_cleaned_langevin))
    # Low-frequency drift below 0.7 Hz should be attenuated
    assert np.sum(fft_raw[freqs < 0.7]) > np.sum(fft_langevin[freqs < 0.7])
    # High-frequency content above 3.5 Hz should be attenuated
    assert np.sum(fft_raw[freqs > 3.5]) > np.sum(fft_langevin[freqs > 3.5])

    # Test that "langevin" alias works identically
    ppg_cleaned_langevin_alias = nk.ppg_clean(ppg, sampling_rate=sampling_rate, method="langevin")
    assert np.allclose(ppg_cleaned_langevin, ppg_cleaned_langevin_alias)


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
    epochs = nk.epochs_create(df, events=[0, 15000], sampling_rate=sampling_rate, epochs_end=150)
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


def test_ppg_findpeaks_all_methods():
    sampling_rate = 100
    duration = 30
    ppg = nk.ppg_simulate(
        duration=duration,
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
        show=False,
    )
    ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=sampling_rate, method="elgendi")

    methods = ["elgendi", "bishop", "charlton", "charlton2024"]
    for method in methods:
        info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate, method=method, show=False)
        assert isinstance(info, dict), f"Failed due to incorrect {method} output type"
        assert "PPG_Peaks" in info, f"Failed as {method} output missing 'PPG_Peaks'"
        peaks = info["PPG_Peaks"]
        assert isinstance(peaks, np.ndarray), f"Failed as {method} peaks of incorrect type"
        assert np.issubdtype(peaks.dtype, np.integer), f"Failed as {method} peaks of incorrect dtype"
        assert np.all((peaks >= 0) & (peaks < len(ppg_cleaned))), f"{method} peaks indices out of range"
        if method in ["charlton", "charlton2024"]:
            assert "PPG_Onsets" in info, f"Failed as {method} missing 'PPG_Onsets'"
            onsets = info["PPG_Onsets"]
            assert isinstance(onsets, np.ndarray), f"Failed as {method} onsets of incorrect type"
            assert np.issubdtype(onsets.dtype, np.integer), f"Failed as {method} onsets of incorrect dtype"
            assert np.all((onsets >= 0) & (onsets < len(ppg_cleaned))), f"Failed as {method} onsets indices out of range"

    with pytest.raises(ValueError):
        nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate, method="invalidmethod")


@pytest.mark.parametrize("sampling_rate", [256, 500, 1000])
def test_ppg_findpeaks_charlton_refines_peaks_and_onsets(sampling_rate):
    ppg = nk.ppg_simulate(duration=30, sampling_rate=sampling_rate, heart_rate=70, random_state=42, show=False)
    ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=sampling_rate)
    info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate, method="charlton", show=False)
    window = int(np.ceil(sampling_rate * 0.15))

    for key, extremum in [("PPG_Peaks", np.argmax), ("PPG_Onsets", np.argmin)]:
        indices = [
            int(index)
            for index in info[key]
            if window <= index < len(ppg_cleaned) - window
        ]
        assert indices
        for index in indices:
            segment = ppg_cleaned[index - window : index + window + 1]
            assert extremum(segment) == window


def test_ppg_peaks_all_methods():
    sampling_rate = 100
    duration = 30
    ppg = nk.ppg_simulate(
        duration=duration,
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
        show=False,
    )
    ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=sampling_rate, method="elgendi")

    methods = ["elgendi", "bishop", "charlton", "charlton2024"]
    for method in methods:
        signals, info = nk.ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate, method=method, show=False)
        assert hasattr(signals, "shape"), f"Failed as {method} signals has no shape"
        assert signals.shape[0] == len(ppg_cleaned), f"Failed as {method} signals length mismatch"
        assert "PPG_Peaks" in signals.columns, f"Failed as {method} signals missing 'PPG_Peaks'"
        assert isinstance(info, dict), f"Failed as {method} info is not a dict"
        assert "PPG_Peaks" in info, f"Failed as {method} info missing 'PPG_Peaks'"
        peaks = info["PPG_Peaks"]
        assert isinstance(peaks, np.ndarray), f"Failed as {method} peaks of incorrect type"
        assert np.issubdtype(peaks.dtype, np.integer), f"Failed as {method} peaks of incorrect dtype"
        assert np.all((peaks >= 0) & (peaks < len(ppg_cleaned))), f"{method} peaks indices out of range"
        if method in ["charlton", "charlton2024"]:
            assert "PPG_Onsets" in info, f"Failed as {method} missing 'PPG_Onsets'"
            onsets = info["PPG_Onsets"]
            assert isinstance(onsets, np.ndarray), f"Failed as {method} onsets of incorrect type"
            assert np.issubdtype(onsets.dtype, np.integer), f"Failed as {method} onsets of incorrect dtype"
            assert np.all((onsets >= 0) & (onsets < len(ppg_cleaned))), f"Failed as {method} onsets indices out of range"

    with pytest.raises(ValueError):
        nk.ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate, method="invalidmethod")


def test_ppg_quality_all_methods():
    sampling_rate = 100
    duration = 120
    ppg_raw = nk.ppg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=70)
    ppg_raw = ppg_raw + np.linspace(10, 30, len(ppg_raw))  # to obtain DC offset
    ppg_cleaned = nk.ppg_clean(ppg_raw, sampling_rate=sampling_rate)
    methods = [
        "templatematch",
        "dissimilarity",
        "ici",
        "skewness",
        "kurtosis",
        "entropy",
        "perfusion",
        "relative_power",
    ]
    for method in methods:
        kwargs = {}
        if method in ["perfusion", "relative_power"]:
            kwargs["ppg_raw"] = ppg_raw
        quality = nk.ppg_quality(ppg_cleaned, sampling_rate=sampling_rate, method=method, **kwargs)
        assert isinstance(quality, (np.ndarray, list)), f"Failed due to incorrect {method} output type"
        assert len(quality) == len(ppg_cleaned), f"Failed due to incorrect {method} output length"
        assert np.all(np.isfinite(quality)), f"Failed due to {method} output containing non-finite values"
        if method == "templatematch":
            assert np.all((quality >= 0) & (quality <= 1)), "Failed as templatematch output should be between 0 and 1"
        if method == "ici":
            assert set(np.unique(quality)).issubset({0, 1}), "Failed as ICI output should be 0 or 1"
        if method == "perfusion":
            assert np.all(quality >= 0), "Failed as perfusion output should be between 0 and 100"


def test_ppg_quality_window_error():
    sampling_rate = 100
    duration = 2  # Shorter than default window for relative_power
    ppg_raw = nk.ppg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=70)
    ppg_cleaned = nk.ppg_clean(ppg_raw, sampling_rate=sampling_rate)
    with pytest.raises(ValueError):
        nk.ppg_quality(
            ppg_cleaned,
            sampling_rate=sampling_rate,
            method="relative_power",
            ppg_raw=ppg_raw,
        )

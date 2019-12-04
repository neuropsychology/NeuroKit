import numpy as np
import pandas as pd
import neurokit2 as nk

# =============================================================================
# Signal
# =============================================================================



def test_signal_resample():

    signal = np.cos(np.linspace(start=0, stop=20, num=50))

    downsampled_fft = nk.signal_resample(signal, method="FFT", sampling_rate=1000, desired_sampling_rate=500)
    downsampled_interpolation = nk.signal_resample(signal, method="interpolation", sampling_rate=1000, desired_sampling_rate=500)

    # Upsample
    upsampled_fft = nk.signal_resample(downsampled_fft, method="FFT", sampling_rate=500, desired_sampling_rate=1000)
    upsampled_interpolation = nk.signal_resample(downsampled_interpolation, method="interpolation", sampling_rate=500, desired_sampling_rate=1000)

    # Check
    rez = pd.DataFrame({"FFT": upsampled_fft - signal,
                        "Interpolation": upsampled_interpolation - signal})
    assert np.allclose(np.mean(rez.mean()), 0.003, atol=0.001)





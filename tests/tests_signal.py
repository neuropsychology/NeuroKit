import numpy as np
import pandas as pd
import neurokit2 as nk

# =============================================================================
# Signal
# =============================================================================



def test_signal_resample():

    signal = np.cos(np.linspace(start=0, stop=20, num=50))

    downsampled_interpolation = nk.signal_resample(signal, method="interpolation", sampling_rate=1000, desired_sampling_rate=500)
    downsampled_interpolation2 = nk.signal_resample(signal, method="interpolation2", sampling_rate=1000, desired_sampling_rate=500)
    downsampled_fft = nk.signal_resample(signal, method="FFT", sampling_rate=1000, desired_sampling_rate=500)
    downsampled_poly = nk.signal_resample(signal, method="poly", sampling_rate=1000, desired_sampling_rate=500)

    # Upsample
    upsampled_interpolation = nk.signal_resample(downsampled_interpolation, method="interpolation", sampling_rate=500, desired_sampling_rate=1000)
    upsampled_interpolation2 = nk.signal_resample(downsampled_interpolation2, method="interpolation2", sampling_rate=500, desired_sampling_rate=1000)
    upsampled_fft = nk.signal_resample(downsampled_fft, method="FFT", sampling_rate=500, desired_sampling_rate=1000)
    upsampled_poly = nk.signal_resample(downsampled_poly, method="poly", sampling_rate=500, desired_sampling_rate=1000)

    # Check
    rez = pd.DataFrame({"Interpolation": upsampled_interpolation - signal,
                        "Interpolation2": upsampled_interpolation2 - signal,
                        "FFT": upsampled_fft - signal,
                        "Poly": upsampled_poly - signal})
    assert np.allclose(np.mean(rez.mean()), -0.0004, atol=0.0001)





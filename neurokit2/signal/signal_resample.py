# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


import neurokit2 as nk


def resample_by_interpolation(signal, input_fs, output_fs):
    """
    Resample a signal by interpolation.

    Parameters
    ----------
    signal :  numpy array
        Array containing the signal.
    input_fs : int, or float
        The original sampling frequency (samples/second).
    output_fs : int, or float
        The target frequency (samples/second).
    
    Returns
    -------
    resampled_signal : numpy array
        Array of the resampled signal values.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> # Downsample
    >>> x = np.linspace(0, 10, 256, endpoint=False)
    >>> y = np.cos(-x**2/6.0)
    >>> yre = resample_by_interpolation(y, 256, 20)
    >>> xre = np.linspace(0, 10, len(yre), endpoint=False)

    >>> plt.figure(figsize=(10, 6))
    >>> plt.plot(x,y,'b', xre,yre,'or-')
    >>> plt.show()

    """
    # calculate new length of sample
    new_length = int(len(signal)*output_fs/input_fs)
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, new_length, endpoint=False),  # where to interpolate
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

def resample_by_FFT(signal, input_fs, output_fs):
    """
    Resample a signal by FFT.
    Parameters
    ----------
    signal : numpy array
        Array containing the signal.
    input_fs : int, or float
        The original sampling frequency (samples/second).
    output_fs : int, or float
        The target frequency (samples/second).
    Returns
    -------
    resampled_signal : numpy array
        Array of the resampled signal values.
    """
    if input_fs == output_fs:
        return signal
    new_length = int(len(signal)*output_fs/input_fs)
    resampled_signal = scipy.signal.resample(signal, new_length)
#    assert len(resampled_signal) == new_length
    return resampled_signal




## TESTING interpolation function
length_difference = []
samples = list(np.linspace(100, 10000, 100, endpoint=False))
for n in samples:
#    print(n)
    x = np.linspace(0, 10, 100, endpoint=False)
    y = np.cos(-x**2/6.0)
    ydown = resample_by_interpolation(y, n, 20)
    yup = resample_by_interpolation(ydown, 20, n) 
    length_difference.append(len(y) - len(yup))

# COMPARE interpolation with scipy.resample
x = np.linspace(0, 10, 256, endpoint=False)
y = np.cos(-x**2/6.0)
plt.plot(x, y)
ydown = resample_by_interpolation(y, 400, 300)

ydown_fft = resample_by_FFT(y, 400, 300)

xdown = np.linspace(0, 10, len(ydown), endpoint=False)

pd.DataFrame({"Interp": ydown, "FFT": ydown_fft}).plot() # plot 2 downsampled

# plot 2 downsampled with original
plt.figure(figsize=(10, 6))
plt.plot(x,y,'b', xdown,ydown,'or-')
plt.plot(xdown, ydown_fft, 'ok-')
plt.legend(['original signal', 'scipy.signal.resample', 'interpolation method'], loc='lower left')
plt.show()



# TEST with ECG signal
bio, bio_sampling_rate= nk.read_acqknowledge("S1.acq", sampling_rate = "max")
#bio = bio.reset_index(drop=True)
ecg = bio["ECG A, X, ECG2-R"]
ecg_down = resample_by_interpolation(ecg, 4000, 3000)
ecg_down_fft = resample_by_FFT(ecg, 4000,3000) 

pd.DataFrame({"Interp": ecg_down, "FFT": ecg_down_fft}).plot()

ecg_up = resample_by_interpolation(ecg_down, 3000, 4000)
ecg_up_fft = resample_by_FFT(ecg_down_fft, 3000, 4000)

pd.DataFrame({"Raw": list(ecg) +[1], "amended": list(ecg_up)}).plot()


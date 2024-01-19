# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np
import pandas as pd
import scipy.signal

from ..misc import NeuroKitWarning, as_vector
from ..signal import signal_detrend, signal_filter
from ..stats import mad


def rsp_clean(rsp_signal, sampling_rate=1000, method="khodadad2018", **kwargs):
    """**Preprocess a respiration (RSP) signal**

    Clean a respiration signal using different sets of parameters, such as:

    * **khodadad2018**: Second order 0.05-3 Hz bandpass Butterworth filter. Note that the
      implementation differs from the referenced paper (see issue #950).
    * **BioSPPy**: Second order 0.1-0.35 Hz bandpass Butterworth filter followed by a constant
      detrending).
    * **hampel**: Applies a median-based Hampel filter by replacing values which are 3 (can be
      changed via ``threshold``) :func:`.mad` away from the rolling median.

    Parameters
    ----------
    rsp_signal : Union[list, np.array, pd.Series]
        The raw respiration channel (as measured, for instance, by a respiration belt).
    sampling_rate : int, optional
        The sampling frequency of :func:`.rsp_signal` (in Hz, i.e., samples/second).
    method : str, optional
        The processing pipeline to apply. Can be one of ``"khodadad2018"`` (default),
        ``"biosppy"`` or ``"hampel"``.
    **kwargs
        Other arguments to pass to the cleaning method.

    Returns
    -------
    array
        Vector containing the cleaned respiratory signal.

    See Also
    --------
    rsp_findpeaks, signal_rate, rsp_amplitude, rsp_process, rsp_plot

    Examples
    --------
    .. ipython:: python

      import pandas as pd
      import neurokit2 as nk

      rsp = nk.rsp_simulate(duration=30, sampling_rate=50, noise=0.1)
      signals = pd.DataFrame({
          "RSP_Raw": rsp,
          "RSP_Khodadad2018": nk.rsp_clean(rsp, sampling_rate=50, method="khodadad2018"),
          "RSP_BioSPPy": nk.rsp_clean(rsp, sampling_rate=50, method="biosppy"),
          "RSP_Hampel": nk.rsp_clean(rsp, sampling_rate=50, method="hampel", threshold=3)
      })
      @savefig p_rsp_clean1.png scale=100%
      signals.plot()
      @suppress
      plt.close()

    References
    ----------
    * Khodadad, D., Nordebo, S., Müller, B., Waldmann, A., Yerworth, R., Becher, T., ... & Bayford,
      R. (2018). Optimized breath detection algorithm in electrical impedance tomography.
      Physiological measurement, 39(9), 094001.
    * Power, J., Lynch, C., Dubin, M., Silver, B., Martin, A., Jones, R.,(2020)
      Characteristics of respiratory measures in young adults scanned at rest,
      including systematic changes and “missed” deep breaths.
      NeuroImage, Volume 204, 116234

    """
    rsp_signal = as_vector(rsp_signal)

    # Missing data
    n_missing = np.sum(np.isnan(rsp_signal))
    if n_missing > 0:
        warn(
            f"There are {n_missing} missing data points in your signal."
            " Filling missing values by using the forward filling method.",
            category=NeuroKitWarning,
        )
        rsp_signal = _rsp_clean_missing(rsp_signal)

    method = method.lower()  # remove capitalised letters
    if method in ["khodadad", "khodadad2018"]:
        clean = _rsp_clean_khodadad2018(rsp_signal, sampling_rate)
    elif method == "biosppy":
        clean = _rsp_clean_biosppy(rsp_signal, sampling_rate)
    elif method in ["power", "power2020", "hampel"]:
        clean = _rsp_clean_hampel(
            rsp_signal,
            **kwargs,
        )
    elif method is None or method == "none":
        clean = rsp_signal
    else:
        raise ValueError(
            "NeuroKit error: rsp_clean(): 'method' should be one of 'khodadad2018', 'biosppy' or 'hampel'."
        )

    return clean


# =============================================================================
# Handle missing data
# =============================================================================
def _rsp_clean_missing(rsp_signal):

    rsp_signal = pd.DataFrame.pad(pd.Series(rsp_signal))

    return rsp_signal


# =============================================================================
# Khodadad et al. (2018)
# =============================================================================
def _rsp_clean_khodadad2018(rsp_signal, sampling_rate=1000):
    """The algorithm is based on (but not an exact implementation of) the "Zero-crossing algorithm with amplitude
    threshold" by `Khodadad et al. (2018)

    <https://iopscience.iop.org/article/10.1088/1361-6579/aad7e6/meta>`_.

    """
    # Slow baseline drifts / fluctuations must be removed from the raw
    # breathing signal (i.e., the signal must be centered around zero) in order
    # to be able to reliable detect zero-crossings.

    # Remove baseline by applying a lowcut at .05Hz (preserves breathing rates
    # higher than 3 breath per minute) and high frequency noise by applying a
    # highcut at 3 Hz (preserves breathing rates slower than 180 breath per
    # minute).
    clean = signal_filter(
        rsp_signal,
        sampling_rate=sampling_rate,
        lowcut=0.05,
        highcut=3,
        order=2,
        method="butterworth",
    )

    return clean


# =============================================================================
# BioSPPy
# =============================================================================
def _rsp_clean_biosppy(rsp_signal, sampling_rate=1000):
    """Uses the same defaults as `BioSPPy.

    <https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/resp.py>`_.

    """
    # Parameters
    order = 2
    frequency = [0.1, 0.35]
    # Normalize frequency to Nyquist Frequency (Fs/2).
    frequency = 2 * np.array(frequency) / sampling_rate

    # Filtering
    b, a = scipy.signal.butter(N=order, Wn=frequency, btype="bandpass", analog=False)
    filtered = scipy.signal.filtfilt(b, a, rsp_signal)

    # Baseline detrending
    clean = signal_detrend(filtered, order=0)

    return clean


# =============================================================================
# Hampel filter
# =============================================================================
def _rsp_clean_hampel(rsp_signal, sampling_rate=1000, window_length=0.1, threshold=3, **kwargs):
    """Explanation MatLabs' https://www.mathworks.com/help/dsp/ref/hampelfilter.html. From
    https://stackoverflow.com/a/51731332.

    Parameters
    ----------
    rsp_signal : Union[list, np.array, pd.Series]
        The raw respiration channel (as measured, for instance, by a respiration belt).
    window_length : int, optional
        Window to be considered when cleaning, by default 0.1. In seconds.
    threshold : float, optional
        Threshold of deviations after which a point is considered an outlier, by default 3.

    """
    # Get window length in samples
    window_length = int(window_length * sampling_rate)

    # Convert to Series to use its rolling methods
    rsp_signal = pd.Series(rsp_signal)

    rolling_median = rsp_signal.rolling(window=window_length, center=True).median()
    rolling_MAD = rsp_signal.rolling(window=window_length, center=True).apply(mad)

    threshold = threshold * rolling_MAD
    difference = np.abs(rsp_signal - rolling_median)
    # Find outliers
    outlier_idx = difference > threshold
    # Substitute outliers with rolling median
    rsp_signal[outlier_idx] = rolling_median[outlier_idx]
    return as_vector(rsp_signal)

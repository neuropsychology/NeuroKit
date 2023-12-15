# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np
import pandas as pd
import scipy.signal

from ..misc import NeuroKitWarning, as_vector
from ..signal import signal_filter


def ecg_clean(ecg_signal, sampling_rate=1000, method="neurokit", **kwargs):
    """**ECG Signal Cleaning**

    Clean an ECG signal to remove noise and improve peak-detection accuracy. Different cleaning
    methods are implemented.

    * ``'neurokit'`` (default): 0.5 Hz high-pass butterworth filter (order = 5), followed by
      powerline filtering (see ``signal_filter()``). By default, ``powerline = 50``.
    * ``'biosppy'``: Method used in the BioSPPy package. A FIR filter ([0.67, 45] Hz; order = 1.5 *
      SR). The 0.67 Hz cutoff value was selected based on the fact that there are no morphological
      features below the heartrate (assuming a minimum heart rate of 40 bpm).
    * ``'pantompkins1985'``: Method used in Pan & Tompkins (1985). **Please help providing a better
      description!**
    * ``'hamilton2002'``: Method used in Hamilton (2002). **Please help providing a better
      description!**
    * ``'elgendi2010'``: Method used in Elgendi et al. (2010). **Please help providing a better
      description!**
    * ``'engzeemod2012'``: Method used in Engelse & Zeelenberg (1979). **Please help providing a
      better description!**
    * ``'vg'``: Method used in Visibility Graph Based Detection Emrich et al. (2023)
      and Koka et al. (2022). A 4.0 Hz high-pass butterworth filter (order = 2).

    Parameters
    ----------
    ecg_signal : Union[list, np.array, pd.Series]
        The raw ECG channel.
    sampling_rate : int
        The sampling frequency of ``ecg_signal`` (in Hz, i.e., samples/second). Defaults to 1000.
    method : str
        The processing pipeline to apply. Can be one of ``"neurokit"`` (default),
        ``"biosppy"``, ``"pantompkins1985"``, ``"hamilton2002"``, ``"elgendi2010"``,
        ``"engzeemod2012"``, ``'vg'``.
    **kwargs
        Other arguments to be passed to specific methods.

    Returns
    -------
    array
        Vector containing the cleaned ECG signal.

    See Also
    --------
    ecg_peaks, ecg_process, ecg_plot, .signal_rate, .signal_filter

    Examples
    --------
    .. ipython:: python

      import pandas as pd
      import numpy as np
      import neurokit2 as nk

      ecg = nk.ecg_simulate(duration=10, sampling_rate=250, noise=0.2)
      ecg += np.random.normal(0, 0.1, len(ecg))  # Add Gaussian noise

      signals = pd.DataFrame({
          "ECG_Raw" : ecg,
          "ECG_NeuroKit" : nk.ecg_clean(ecg, sampling_rate=250, method="neurokit"),
          "ECG_BioSPPy" : nk.ecg_clean(ecg, sampling_rate=250, method="biosppy"),
          "ECG_PanTompkins" : nk.ecg_clean(ecg, sampling_rate=250, method="pantompkins1985"),
          "ECG_Hamilton" : nk.ecg_clean(ecg, sampling_rate=250, method="hamilton2002"),
          "ECG_Elgendi" : nk.ecg_clean(ecg, sampling_rate=250, method="elgendi2010"),
          "ECG_EngZeeMod" : nk.ecg_clean(ecg, sampling_rate=250, method="engzeemod2012"),
          "ECG_VG" : nk.ecg_clean(ecg, sampling_rate=250, method="vg"),
          "ECG_TC" : nk.ecg_clean(ecg, sampling_rate=250, method="templateconvolution")
      })

      @savefig p_ecg_clean.png scale=100%
      signals.plot(subplots=True)
      @suppress
      plt.close()


    References
    --------------
    * Engelse, W. A., & Zeelenberg, C. (1979). A single scan algorithm for QRS-detection and
      feature extraction. Computers in cardiology, 6(1979), 37-42.
    * Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE transactions
      on biomedical engineering, (3), 230-236.
    * Hamilton, P. (2002). Open source ECG analysis. In Computers in cardiology (pp. 101-104).
      IEEE.
    * Elgendi, M., Jonkman, M., & De Boer, F. (2010). Frequency Bands Effects on QRS Detection.
      Biosignals, Proceedings of the Third International Conference on Bio-inspired Systems and
      Signal Processing, 428-431.
    * Emrich, J., Koka, T., Wirth, S., & Muma, M. (2023), Accelerated Sample-Accurate R-Peak
      Detectors Based on Visibility Graphs. 31st European Signal Processing Conference
      (EUSIPCO), 1090-1094, doi: 10.23919/EUSIPCO58844.2023.10290007

    """
    ecg_signal = as_vector(ecg_signal)

    # Missing data
    n_missing = np.sum(np.isnan(ecg_signal))
    if n_missing > 0:
        warn(
            "There are " + str(n_missing) + " missing data points in your signal."
            " Filling missing values by using the forward filling method.",
            category=NeuroKitWarning,
        )
        ecg_signal = _ecg_clean_missing(ecg_signal)

    method = method.lower()  # remove capitalised letters
    if method in ["nk", "nk2", "neurokit", "neurokit2"]:
        clean = _ecg_clean_nk(ecg_signal, sampling_rate, **kwargs)
    elif method in ["biosppy", "gamboa2008"]:
        clean = _ecg_clean_biosppy(ecg_signal, sampling_rate)
    elif method in ["pantompkins", "pantompkins1985"]:
        clean = _ecg_clean_pantompkins(ecg_signal, sampling_rate)
    elif method in ["hamilton", "hamilton2002"]:
        clean = _ecg_clean_hamilton(ecg_signal, sampling_rate)
    elif method in ["elgendi", "elgendi2010"]:
        clean = _ecg_clean_elgendi(ecg_signal, sampling_rate)
    elif method in ["engzee", "engzee2012", "engzeemod", "engzeemod2012"]:
        clean = _ecg_clean_engzee(ecg_signal, sampling_rate)
    elif method in ["vg", "vgraph", "fastnvg", "emrich", "emrich2023"]:
        clean = _ecg_clean_vgraph(ecg_signal, sampling_rate)
    elif method in ["koka2022", "koka"]:
        warn(
            "The 'koka2022' method has been replaced by 'emrich2023'."
            " Please replace method='koka2022' by method='emrich2023'.",
            category=NeuroKitWarning,
        )
        clean = _ecg_clean_vgraph(ecg_signal, sampling_rate)
    elif method in ["templateconvolution"]:
        clean = _ecg_clean_templateconvolution(ecg_signal, sampling_rate)
    elif method in [
        "christov",
        "christov2004",
        "ssf",
        "slopesumfunction",
        "zong",
        "zong2003",
        "kalidas2017",
        "swt",
        "kalidas",
        "kalidastamil",
        "kalidastamil2017",
    ]:
        clean = ecg_signal
    else:
        raise ValueError(
            "NeuroKit error: ecg_clean(): 'method' should be "
            "one of 'neurokit', 'biosppy', 'pantompkins1985',"
            " 'hamilton2002', 'elgendi2010', 'engzeemod2012',"
            " 'templateconvolution', 'vg'."
        )
    return clean


# =============================================================================
# Handle missing data
# =============================================================================
def _ecg_clean_missing(ecg_signal):
    ecg_signal = pd.DataFrame.pad(pd.Series(ecg_signal))

    return ecg_signal


# =============================================================================
# NeuroKit
# =============================================================================
def _ecg_clean_nk(ecg_signal, sampling_rate=1000, **kwargs):
    # Remove slow drift and dc offset with highpass Butterworth.
    clean = signal_filter(
        signal=ecg_signal,
        sampling_rate=sampling_rate,
        lowcut=0.5,
        method="butterworth",
        order=5,
    )

    clean = signal_filter(signal=clean, sampling_rate=sampling_rate, method="powerline", **kwargs)
    return clean


# =============================================================================
# Biosppy
# =============================================================================
def _ecg_clean_biosppy(ecg_signal, sampling_rate=1000):
    """Adapted from https://github.com/PIA-
    Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L69.
    """

    # The order and frequency was recently changed
    # (see https://github.com/scientisst/BioSPPy/pull/12)

    order = int(1.5 * sampling_rate)
    if order % 2 == 0:
        order += 1  # Enforce odd number

    # -> filter_signal()
    frequency = [0.67, 45]

    #   -> get_filter()
    #     -> _norm_freq()
    frequency = 2 * np.array(frequency) / sampling_rate  # Normalize frequency to Nyquist Frequency (Fs/2).

    #     -> get coeffs
    a = np.array([1])
    b = scipy.signal.firwin(numtaps=order, cutoff=frequency, pass_zero=False)

    # _filter_signal()
    filtered = scipy.signal.filtfilt(b, a, ecg_signal)

    # DC offset
    filtered -= np.mean(filtered)

    return filtered


# =============================================================================
# Pan & Tompkins (1985)
# =============================================================================
def _ecg_clean_pantompkins(ecg_signal, sampling_rate=1000):
    """Adapted from https://github.com/PIA-
    Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L69.
    """

    order = 1
    clean = signal_filter(
        signal=ecg_signal,
        sampling_rate=sampling_rate,
        lowcut=5,
        highcut=15,
        method="butterworth_zi",
        order=order,
    )

    return clean  # Return filtered


# =============================================================================
# Hamilton (2002)
# =============================================================================
def _ecg_clean_hamilton(ecg_signal, sampling_rate=1000):
    """Adapted from https://github.com/PIA-
    Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L69.
    """

    order = 1
    clean = signal_filter(
        signal=ecg_signal,
        sampling_rate=sampling_rate,
        lowcut=8,
        highcut=16,
        method="butterworth_zi",
        order=order,
    )

    return clean  # Return filtered


# =============================================================================
# Elgendi et al. (2010)
# =============================================================================
def _ecg_clean_elgendi(ecg_signal, sampling_rate=1000):
    """From https://github.com/berndporr/py-ecg-detectors/

    - Elgendi, Mohamed & Jonkman, Mirjam & De Boer, Friso. (2010). Frequency Bands Effects on QRS
      Detection. The 3rd International Conference on Bio-inspired Systems and Signal Processing
      (BIOSIGNALS2010). 428-431.

    """

    order = 2
    clean = signal_filter(
        signal=ecg_signal,
        sampling_rate=sampling_rate,
        lowcut=8,
        highcut=20,
        method="butterworth_zi",
        order=order,
    )

    return clean  # Return filtered


# =============================================================================
# Engzee Modified (2012)
# =============================================================================
def _ecg_clean_engzee(ecg_signal, sampling_rate=1000):
    """From https://github.com/berndporr/py-ecg-detectors/

    - C. Zeelenberg, A single scan algorithm for QRS detection and feature extraction, IEEE Comp.
      in Cardiology, vol. 6, pp. 37-42, 1979.
    - A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred, "Real Time Electrocardiogram
      Segmentation for Finger Based ECG Biometrics", BIOSIGNALS 2012, pp. 49-54, 2012.

    """

    order = 4
    clean = signal_filter(
        signal=ecg_signal,
        sampling_rate=sampling_rate,
        lowcut=52,
        highcut=48,
        method="butterworth_zi",
        order=order,
    )

    return clean  # Return filtered


# =============================================================================
# Visibility-Graph Detector - Emrich et al. (2023) & Koka et al. (2022)
# =============================================================================
def _ecg_clean_vgraph(ecg_signal, sampling_rate=1000):
    """Filtering used for Visibility-Graph Detectors Emrich et al. (2023) and Koka et al. (2022).

    - J. Emrich, T. Koka, S. Wirth and M. Muma, "Accelerated Sample-Accurate R-Peak
      Detectors Based on Visibility Graphs," 31st European Signal Processing
      Conference (EUSIPCO), 2023, pp. 1090-1094, doi: 10.23919/EUSIPCO58844.2023.10290007,
      https://ieeexplore.ieee.org/document/10290007

    - T. Koka and M. Muma (2022), Fast and Sample Accurate R-Peak Detection for Noisy ECG Using
      Visibility Graphs. In: 2022 44th Annual International Conference of the IEEE Engineering
      in Medicine & Biology Society (EMBC). Uses the Pan and Tompkins thresholding.

    """

    order = 2
    clean = signal_filter(
        signal=ecg_signal,
        sampling_rate=sampling_rate,
        lowcut=4,
        method="butterworth",
        order=order,
    )

    return clean  # Return filtered


# =============================================================================
# Template Convolution (Exploratory)
# =============================================================================
def _ecg_clean_templateconvolution(ecg_signal, sampling_rate=1000):
    """Filter and Convolve ECG signal with QRS complex template. Totally exploratory method by Dominique Makowski, use
    at your own risks.

    The idea is to use a QRS template to convolve the signal with, in order to magnify the QRS features. However,
    it doens't work well and creates a lot of artifacts. If you have ideas for improvement please let me know!

    """

    window_size = int(np.round(sampling_rate / 4))
    if (window_size % 2) == 0:
        window_size + 1

    # Filter out slow drifts and high freq noises
    filtered = signal_filter(
        signal=ecg_signal,
        sampling_rate=sampling_rate,
        lowcut=1,
        highcut=40,
        method="butterworth",
        window_size=window_size,
    )

    # Detect peaks
    peaks, _ = scipy.signal.find_peaks(filtered, distance=sampling_rate / 3, height=0.5 * np.std(filtered))
    peaks = peaks[peaks + 0.6 * sampling_rate < len(ecg_signal)]

    idx = [np.arange(p - int(sampling_rate / 2), p + int(sampling_rate / 2)) for p in peaks]
    epochs = np.array([filtered[i] for i in idx])
    qrs = np.mean(epochs, axis=0)

    # # Create base QRS template using wavelets
    # qrs = scipy.signal.ricker(600, a=16)

    # # Adjust template
    # qrs[100:220] = qrs[100:220] + 0.03 * np.sin(np.linspace(0, 1 * np.pi, 120))
    # qrs[284:316] = qrs[284:316] + 0.2 * qrs[284:316]
    # qrs[316:380] = qrs[316:380] + 0.5 * qrs[316:380]
    # qrs[380:500] = qrs[380:500] + 0.05 * np.sin(np.linspace(0, 1 * np.pi, 120))

    # # Resample
    # qrs = signal_resample(qrs, desired_length=sampling_rate)

    return scipy.signal.convolve(ecg_signal, qrs, mode="same")

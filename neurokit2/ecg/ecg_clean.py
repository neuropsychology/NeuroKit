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
    method are implemented.

    * ``'neurokit'`` (default): 0.5 Hz high-pass butterworth filter (order = 5), followed by
      powerline filtering (see ``signal_filter()``). By default, ``powerline = 50``.
    * ``'biosppy'``: Same as in the biosppy package. **Please help providing a better description!**
    * ``'pantompkins1985'``: Method used in Pan & Tompkins (1985). **Please help providing a better
      description!**
    * ``'hamilton2002'``: Method used in Hamilton (2002). **Please help providing a better
      description!**
    * ``'elgendi2010'``: Method used in Elgendi et al. (2010). **Please help providing a better
      description!**
    * ``'engzeemod2012'``: Method used in Engelse & Zeelenberg (1979). **Please help providing a
      better description!**


    Parameters
    ----------
    ecg_signal : Union[list, np.array, pd.Series]
        The raw ECG channel.
    sampling_rate : int
        The sampling frequency of ``ecg_signal`` (in Hz, i.e., samples/second). Defaults to 1000.
    method : str
        The processing pipeline to apply. Can be one of ``"neurokit"`` (default),
        ``"biosppy"``, ``"pantompkins1985"``, ``"hamilton2002"``, ``"elgendi2010"``,
        ``"engzeemod2012"``.
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
      import neurokit2 as nk
      import matplotlib.pyplot as plt

      ecg = nk.ecg_simulate(duration=10, sampling_rate=1000)
      signals = pd.DataFrame({"ECG_Raw" : ecg,
                              "ECG_NeuroKit" : nk.ecg_clean(ecg, sampling_rate=1000, method="neurokit"),
                              "ECG_BioSPPy" : nk.ecg_clean(ecg, sampling_rate=1000, method="biosppy"),
                              "ECG_PanTompkins" : nk.ecg_clean(ecg, sampling_rate=1000, method="pantompkins1985"),
                              "ECG_Hamilton" : nk.ecg_clean(ecg, sampling_rate=1000, method="hamilton2002"),
                              "ECG_Elgendi" : nk.ecg_clean(ecg, sampling_rate=1000, method="elgendi2010"),
                              "ECG_EngZeeMod" : nk.ecg_clean(ecg, sampling_rate=1000, method="engzeemod2012")})
      @savefig p_ecg_clean.png scale=100%
      signals.plot()


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
    elif method in ["vg", "vgraph", "koka2022"]:
        clean = _ecg_clean_vgraph(ecg_signal, sampling_rate)
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
        "kalidastamil2017"
    ]:
        clean = ecg_signal
    else:
        raise ValueError(
            "NeuroKit error: ecg_clean(): 'method' should be "
            "one of 'neurokit', 'biosppy', 'pantompkins1985',"
            " 'hamilton2002', 'elgendi2010', 'engzeemod2012'."
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
    clean = signal_filter(signal=ecg_signal, sampling_rate=sampling_rate, lowcut=0.5, method="butterworth", order=5)

    clean = signal_filter(signal=clean, sampling_rate=sampling_rate, method="powerline", **kwargs)
    return clean


# =============================================================================
# Biosppy
# =============================================================================
def _ecg_clean_biosppy(ecg_signal, sampling_rate=1000):
    """Adapted from https://github.com/PIA-
    Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L69."""

    order = int(0.3 * sampling_rate)
    if order % 2 == 0:
        order += 1  # Enforce odd number

    # -> filter_signal()
    frequency = [3, 45]

    #   -> get_filter()
    #     -> _norm_freq()
    frequency = 2 * np.array(frequency) / sampling_rate  # Normalize frequency to Nyquist Frequency (Fs/2).

    #     -> get coeffs
    a = np.array([1])
    b = scipy.signal.firwin(numtaps=order, cutoff=frequency, pass_zero=False)

    # _filter_signal()
    filtered = scipy.signal.filtfilt(b, a, ecg_signal)

    return filtered


# =============================================================================
# Pan & Tompkins (1985)
# =============================================================================
def _ecg_clean_pantompkins(ecg_signal, sampling_rate=1000):
    """Adapted from https://github.com/PIA-
    Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L69."""

    order = 1
    clean = signal_filter(
        signal=ecg_signal, sampling_rate=sampling_rate, lowcut=5, highcut=15, method="butterworth_zi", order=order
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
        signal=ecg_signal, sampling_rate=sampling_rate, lowcut=8, highcut=20, method="butterworth_zi", order=order
    )

    return clean  # Return filtered


# =============================================================================
# Hamilton (2002)
# =============================================================================
def _ecg_clean_hamilton(ecg_signal, sampling_rate=1000):
    """Adapted from https://github.com/PIA-
    Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L69."""

    order = 1
    clean = signal_filter(
        signal=ecg_signal, sampling_rate=sampling_rate, lowcut=8, highcut=16, method="butterworth_zi", order=order
    )

    return clean  # Return filtered


# =============================================================================
# Engzee Modified (2012)
# =============================================================================
def _ecg_clean_engzee(ecg_signal, sampling_rate=1000):
    """From https://github.com/berndporr/py-ecg-detectors/

    - C. Zeelenberg, A single scan algorithm for QRS detection and feature extraction, IEEE Comp.
      in Cardiology, vol. 6, pp. 37-42, 1979.

    - A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred, "Real Time Electrocardiogram Segmentation
      for Finger Based ECG Biometrics", BIOSIGNALS 2012, pp. 49-54, 2012.

    """

    order = 4
    clean = signal_filter(
        signal=ecg_signal, sampling_rate=sampling_rate, lowcut=52, highcut=48, method="butterworth_zi", order=order
    )

    return clean  # Return filtered


# =============================================================================
# Engzee Modified (2012)
# =============================================================================
def _ecg_clean_vgraph(ecg_signal, sampling_rate=1000):
    """Filtering used by Taulant Koka and Michael Muma (2022).

    References
    ----------
    - T. Koka and M. Muma (2022), Fast and Sample Accurate R-Peak Detection for Noisy ECG Using
      Visibility Graphs. In: 2022 44th Annual International Conference of the IEEE Engineering
      in Medicine & Biology Society (EMBC). Uses the Pan and Tompkins thresholding.

    """

    order = 2
    clean = signal_filter(signal=ecg_signal, sampling_rate=sampling_rate, lowcut=4, method="butterworth", order=order)

    return clean  # Return filtered

# -*- coding: utf-8 -*-
import pandas as pd

from ..signal import signal_rate, signal_sanitize
from .ecg_clean import ecg_clean
from .ecg_delineate import ecg_delineate
from .ecg_peaks import ecg_peaks
from .ecg_phase import ecg_phase
from .ecg_quality import ecg_quality


def ecg_process(ecg_signal, sampling_rate=1000, method="neurokit", **kwargs):
    """**Automated pipeline for preprocessing an ECG signal**

    This function runs different preprocessing steps: Cleaning (using ``ecg_clean()``),
    peak detection (using ``ecg_peaks()``), heart rate calculation (using ``signal_rate()``),
    signal quality assessment (using ``ecg_quality()``),
    QRS complex delineation (using ``ecg_delineate()``),
    and cardiac phase determination (using ``ecg_phase()``).

    **Help us improve the documentation of this function by making it more tidy and useful!**

    Parameters
    ----------
    ecg_signal : Union[list, np.array, pd.Series]
        The raw single-channel ECG signal.
    sampling_rate : int
        The sampling frequency of ``ecg_signal`` (in Hz, i.e., samples/second). Defaults to 1000.
    method : str
        The processing method used for signal cleaning (using ``ecg_clean()``) and peak detection
        (using ``ecg_peaks()``). Defaults to ``'neurokit'``. Available methods are ``'neurokit'``,
        ``'pantompkins1985'``, ``'hamilton2002'``, ``'elgendi2010'``, ``'engzeemod2012'``.
        We aim at improving this aspect to make the available methods more transparent, and be able
        to generate specific reports. Please get in touch if you are interested in helping out with
        this.

    Returns
    -------
    signals : DataFrame
        A DataFrame of the same length as the ``ecg_signal`` containing the following columns:

        .. codebookadd::
            ECG_Raw|The raw signal.
            ECG_Clean|The cleaned signal.
            ECG_Rate|Heart rate interpolated between R-peaks.
            ECG_Quality|The quality of the cleaned signal.
            ECG_R_Peaks|The R-peaks marked as "1" in a list of zeros.
            ECG_R_Onsets|The R-onsets marked as "1" in a list of zeros.
            ECG_R_Offsets|The R-offsets marked as "1" in a list of zeros.
            ECG_P_Peaks|The P-peaks marked as "1" in a list of zeros.
            ECG_P_Onsets|The P-onsets marked as "1" in a list of zeros.
            ECG_P_Offsets|The P-offsets marked as "1" in a list of zeros.
            ECG_Q_Peaks|The Q-peaks marked as "1" in a list of zeros.
            ECG_S_Peaks|The S-peaks marked as "1" in a list of zeros.
            ECG_T_Peaks|The T-peaks marked as "1" in a list of zeros.
            ECG_T_Onsets|The T-onsets marked as "1" in a list of zeros.
            ECG_T_Offsets|The T-offsets marked as "1" in a list of zeros.
            ECG_Phase_Atrial|Cardiac phase, marked by "1" for systole and "0" for diastole.
            ECG_Phase_Completion_Atrial|Cardiac phase (atrial) completion, expressed in \
                percentage (from 0 to 1), representing the stage of the current cardiac phase.
            ECG_Phase_Completion_Ventricular|Cardiac phase (ventricular) completion, expressed \
                in percentage (from 0 to 1), representing the stage of the current cardiac phase.

    rpeaks : dict
        A dictionary containing the samples at which the R-peaks occur, accessible with the key
        ``"ECG_R_Peaks"``, as well as the signals' sampling rate.

    See Also
    --------
    ecg_clean, ecg_peaks, ecg_quality, ecg_delineate, ecg_phase, ecg_plot, .signal_rate

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate ECG signal
      ecg = nk.ecg_simulate(duration=15, sampling_rate=1000, heart_rate=80)

      # Preprocess ECG signal
      signals, info = nk.ecg_process(ecg, sampling_rate=1000)

      # Visualize
      @savefig p_ecg_process.png scale=100%
      nk.ecg_plot(signals, info)
      @suppress
      plt.close()



    """

    # Sanitize and clean input
    ecg_signal = signal_sanitize(ecg_signal)
    ecg_cleaned = ecg_clean(ecg_signal, sampling_rate=sampling_rate, method=method, **kwargs)

    # Detect R-peaks
    instant_peaks, info = ecg_peaks(
        ecg_cleaned=ecg_cleaned,
        sampling_rate=sampling_rate,
        method=method,
        correct_artifacts=True,
    )

    # Calculate heart rate
    rate = signal_rate(
        info, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned)
    )

    # Assess signal quality
    quality = ecg_quality(
        ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate
    )

    # Merge signals in a DataFrame
    signals = pd.DataFrame(
        {
            "ECG_Raw": ecg_signal,
            "ECG_Clean": ecg_cleaned,
            "ECG_Rate": rate,
            "ECG_Quality": quality,
        }
    )

    # Delineate QRS complex
    delineate_signal, delineate_info = ecg_delineate(
        ecg_cleaned=ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate
    )
    info.update(delineate_info)  # Merge waves indices dict with info dict

    # Determine cardiac phases
    cardiac_phase = ecg_phase(
        ecg_cleaned=ecg_cleaned,
        rpeaks=info["ECG_R_Peaks"],
        delineate_info=delineate_info,
    )

    # Add additional information to signals DataFrame
    signals = pd.concat(
        [signals, instant_peaks, delineate_signal, cardiac_phase], axis=1
    )

    # return signals DataFrame and R-peak locations
    return signals, info

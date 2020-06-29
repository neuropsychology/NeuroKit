# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from ..signal import signal_filter, signal_findpeaks, signal_smooth, signal_zerocrossings


def eda_findpeaks(eda_phasic, sampling_rate=1000, method="neurokit", amplitude_min=0.1):
    """Identify Skin Conductance Responses (SCR) in Electrodermal Activity (EDA).

    Low-level function used by `eda_peaks()` to identify Skin Conductance Responses (SCR) peaks in the
    phasic component of Electrodermal Activity (EDA) with different possible methods. See `eda_peaks()` for details.

    Parameters
    ----------
    eda_phasic : Union[list, np.array, pd.Series]
        The phasic component of the EDA signal (from `eda_phasic()`).
    sampling_rate : int
        The sampling frequency of the EDA signal (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of "neurokit" (default),
        "gamboa2008", "kim2004" (the default in BioSPPy) or "vanhalem2020".
    amplitude_min : float
        Only used if 'method' is 'neurokit' or 'kim2004'. Minimum threshold by which to exclude
        SCRs (peaks) as relative to the largest amplitude in the signal.

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the aplitude of the SCR, the samples
        at which the SCR onset and the SCR peaks occur. Accessible with the keys "SCR_Amplitude",
        "SCR_Onsets", and "SCR_Peaks" respectively.

    See Also
    --------
    eda_simulate, eda_clean, eda_phasic, eda_fixpeaks, eda_peaks, eda_process, eda_plot



    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> # Get phasic component
    >>> eda_signal = nk.eda_simulate(duration=30, scr_number=5, drift=0.1, noise=0)
    >>> eda_cleaned = nk.eda_clean(eda_signal)
    >>> eda = nk.eda_phasic(eda_cleaned)
    >>> eda_phasic = eda["EDA_Phasic"].values
    >>>
    >>> # Find peaks
    >>> gamboa2008 = nk.eda_findpeaks(eda_phasic, method="gamboa2008")
    >>> kim2004 = nk.eda_findpeaks(eda_phasic, method="kim2004")
    >>> neurokit = nk.eda_findpeaks(eda_phasic, method="neurokit")
    >>> vanhalem2020 = nk.eda_findpeaks(eda_phasic, method="vanhalem2020")
    >>> fig = nk.events_plot([gamboa2008["SCR_Peaks"], kim2004["SCR_Peaks"], vanhalem2020["SCR_Peaks"],
    ...                       neurokit["SCR_Peaks"]], eda_phasic)
    >>> fig #doctest: +SKIP

    References
    ----------
    - Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci and electrophysiology.
      PhD ThesisUniversidade.

    - Kim, K. H., Bang, S. W., & Kim, S. R. (2004). Emotion recognition system using short-term monitoring
      of physiological signals. Medical and biological engineering and computing, 42(3), 419-427.

    - van Halem, S., Van Roekel, E., Kroencke, L., Kuper, N., & Denissen, J. (2020).
      Moments That Matter? On the Complexity of Using Triggers Based on Skin Conductance to Sample
      Arousing Events Within an Experience Sampling Framework. European Journal of Personality.

    """
    # Try to retrieve the right column if a dataframe is passed
    if isinstance(eda_phasic, pd.DataFrame):
        try:
            eda_phasic = eda_phasic["EDA_Phasic"]
        except KeyError:
            raise KeyError("NeuroKit error: eda_findpeaks(): Please provide an array as the input signal.")

    method = method.lower()  # remove capitalised letters
    if method in ["gamboa2008", "gamboa"]:
        info = _eda_findpeaks_gamboa2008(eda_phasic)
    elif method in ["kim", "kbk", "kim2004", "biosppy"]:
        info = _eda_findpeaks_kim2004(eda_phasic, sampling_rate=sampling_rate, amplitude_min=amplitude_min)
    elif method in ["nk", "nk2", "neurokit", "neurokit2"]:
        info = _eda_findpeaks_neurokit(eda_phasic, amplitude_min=amplitude_min)
    elif method in ["vanhalem2020", "vanhalem", "halem2020"]:
        info = _eda_findpeaks_vanhalem2020(eda_phasic, sampling_rate=sampling_rate)
    else:
        raise ValueError(
            "NeuroKit error: eda_findpeaks(): 'method' should be one of 'neurokit', 'gamboa2008', 'kim2004'"
            " or 'vanhalem2020'."
        )

    return info


# =============================================================================
# Methods
# =============================================================================


def _eda_findpeaks_neurokit(eda_phasic, amplitude_min=0.1):

    peaks = signal_findpeaks(eda_phasic, relative_height_min=amplitude_min, relative_max=True)

    info = {"SCR_Onsets": peaks["Onsets"], "SCR_Peaks": peaks["Peaks"], "SCR_Height": eda_phasic[peaks["Peaks"]]}

    return info


def _eda_findpeaks_vanhalem2020(eda_phasic, sampling_rate=1000):
    """Follows approach of van Halem et al. (2020).

    A peak is considered when there is a consistent increase of 0.5 seconds following a consistent decrease
    of 0.5 seconds.

    Parameters
    ----------
    eda_phasic : array
        Input filterd EDA signal.
    sampling_rate : int
        Sampling frequency (Hz). Defaults to 1000Hz.

    Returns
    -------
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SRC peaks.
    amplitudes : array
        SCR pulse amplitudes.

    References
    ----------
    - van Halem, S., Van Roekel, E., Kroencke, L., Kuper, N., & Denissen, J. (2020).
    Moments That Matter? On the Complexity of Using Triggers Based on Skin Conductance to Sample
    Arousing Events Within an Experience Sampling Framework. European Journal of Personality.

    """
    # smooth
    eda_phasic = signal_filter(
        eda_phasic, sampling_rate=sampling_rate, lowcut=None, highcut=None, method="savgol", window_size=501
    )
    info = signal_findpeaks(eda_phasic)
    peaks = info["Peaks"]

    threshold = 0.5 * sampling_rate

    # Define each peak as a consistent increase of 0.5s
    peaks = peaks[info["Width"] > threshold]
    idx = np.where(peaks[:, None] == info["Peaks"][None, :])[1]

    # Check if each peak is followed by consistent decrease of 0.5s
    decrease = info["Offsets"][idx] - peaks
    if any(np.isnan(decrease)):
        decrease[np.isnan(decrease)] = False
    if any(decrease < threshold):
        keep = np.where(decrease > threshold)[0]
        idx = idx[keep]  # Update index

    info = {"SCR_Onsets": info["Onsets"][idx], "SCR_Peaks": info["Peaks"][idx], "SCR_Height": info["Height"][idx]}

    return info


def _eda_findpeaks_gamboa2008(eda_phasic):
    """Basic method to extract Skin Conductivity Responses (SCR) from an EDA signal following the approach in the thesis
    by Gamboa (2008).

    Parameters
    ----------
    eda_phasic : array
        Input filterd EDA signal.

    Returns
    -------
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SRC peaks.
    amplitudes : array
        SCR pulse amplitudes.

    References
    ----------
    - Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci and electrophysiology.
      PhD ThesisUniversidade.

    """
    derivative = np.diff(np.sign(np.diff(eda_phasic)))

    # find extrema
    pi = np.nonzero(derivative < 0)[0] + 1
    ni = np.nonzero(derivative > 0)[0] + 1

    # sanity check
    if len(pi) == 0 or len(ni) == 0:
        raise ValueError("NeuroKit error: eda_findpeaks(): Could not find enough SCR peaks. Try another method.")

    # pair vectors
    if ni[0] < pi[0]:
        ni = ni[1:]
    if pi[-1] > ni[-1]:
        pi = pi[:-1]
    if len(pi) > len(ni):
        pi = pi[:-1]

    li = min(len(pi), len(ni))
    peaks = pi[:li]
    onsets = ni[:li]

    # indices
    i0 = peaks - (onsets - peaks) / 2.0
    if i0[0] < 0:
        i0[0] = 0

    # amplitude
    amplitudes = np.array([np.max(eda_phasic[peaks[i] : onsets[i]]) for i in range(li)])

    # output
    info = {"SCR_Onsets": onsets, "SCR_Peaks": peaks, "SCR_Height": amplitudes}
    return info


def _eda_findpeaks_kim2004(eda_phasic, sampling_rate=1000, amplitude_min=0.1):
    """KBK method to extract Skin Conductivity Responses (SCR) from an EDA signal following the approach by Kim et
    al.(2004).

    Parameters
    ----------
    eda_phasic : array
        Input filterd EDA signal.
    sampling_rate : int
        Sampling frequency (Hz). Defaults to 1000Hz.
    amplitude_min : float
        Minimum treshold by which to exclude SCRs. Defaults to 0.1.

    Returns
    -------
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SRC peaks.
    amplitudes : array
        SCR pulse amplitudes.

    References
    ----------
    - Kim, K. H., Bang, S. W., & Kim, S. R. (2004). Emotion recognition system using short-term monitoring
      of physiological signals. Medical and biological engineering and computing, 42(3), 419-427.

    """

    # differentiation
    df = np.diff(eda_phasic)

    # smooth
    df = signal_smooth(signal=df, kernel="bartlett", size=int(sampling_rate))

    # zero crosses
    zeros = signal_zerocrossings(df)
    if np.all(df[: zeros[0]] > 0):
        zeros = zeros[1:]
    if np.all(df[zeros[-1] :] > 0):
        zeros = zeros[:-1]

    # exclude SCRs with small amplitude
    thr = amplitude_min * np.max(df)

    scrs, amps, ZC, pks = [], [], [], []
    for i in range(0, len(zeros) - 1, 2):
        scrs += [df[zeros[i] : zeros[i + 1]]]
        aux = scrs[-1].max()
        if aux > thr:
            amps += [aux]
            ZC += [zeros[i]]
            ZC += [zeros[i + 1]]
            pks += [zeros[i] + np.argmax(df[zeros[i] : zeros[i + 1]])]

    scrs = np.array(scrs)
    amps = np.array(amps)
    ZC = np.array(ZC)
    pks = np.array(pks)
    onsets = ZC[::2]

    # output
    info = {"SCR_Onsets": onsets, "SCR_Peaks": pks, "SCR_Height": amps}

    return info

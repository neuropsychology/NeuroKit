# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

from ..signal import signal_smooth
from ..signal import signal_zerocrossings



def eda_findpeaks(eda_phasic, sampling_rate=1000, method="gamboa2008"):
    """Decompose Electrodermal Activity (EDA) into Phasic and Tonic components.

    Decompose the Electrodermal Activity (EDA) into two components, namely Phasic and Tonic, using different methods including cvxEDA (Greco, 2016) or Biopac's Acqknowledge algorithms.

    Parameters
    ----------
    eda_signal : list, array or Series
        The raw EDA signal.
    sampling_rate : int
        The sampling frequency of `rsp_signal` (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of "cvxEDA"
        (default) or "biosppy".

    Returns
    -------
    DataFrame
        DataFrame containing the 'Tonic' and the 'Phasic' components as columns.

    See Also
    --------
    eda_simulate, eda_clean, eda_decompose



    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> # Get phasic component
    >>> eda_signal = nk.eda_simulate(duration=30, n_scr=5, drift=0.1, noise=0)
    >>> eda_cleaned = nk.eda_clean(eda_signal)
    >>> eda = nk.eda_decompose(eda_cleaned)
    >>> eda_phasic = eda["EDA_Phasic"]
    >>>
    >>> # Find peaks
    >>> signals, info_gamboa2008 = nk.eda_findpeaks(eda_phasic, method="gamboa2008")
    >>> signals, info_kim2004 = nk.eda_findpeaks(eda_phasic, method="kim2004")
    >>> nk.events_plot([info_gamboa2008["SCR_Peaks"], info_kim2004["SCR_Peaks"]], eda_phasic)
    """
    method = method.lower()  # remove capitalised letters
    if method in ["gamboa2008", "gamboa"]:
        info = _eda_findpeaks_gamboa2008(eda_phasic)
    elif method in ["kim", "kbk", "kim2004"]:
        info = _eda_findpeaks_kim2004(eda_phasic, sampling_rate=sampling_rate, amplitude_min=0.1)
    else:
        raise ValueError("NeuroKit error: eda_findpeaks(): 'method' should be "
                         "one of 'gamboa2008'.")

    # Prepare output.
    peaks_signal = np.zeros(len(eda_phasic))
    peaks_signal[info["SCR_Peaks"]] = 1
    signals = pd.DataFrame({"SCR_Peaks": peaks_signal})

    return signals, info




# =============================================================================
# Methods
# =============================================================================



def _eda_findpeaks_gamboa2008(eda_phasic):
    """Basic method to extract Skin Conductivity Responses (SCR) from an
    EDA signal following the approach in the thesis by Gamboa (2008).

    References
    ----------
    - Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci and electrophysiology. PhD ThesisUniversidade.
    """
    derivative = np.diff(np.sign(np.diff(eda_phasic)))

    # find extrema
    pi = np.nonzero(derivative < 0)[0] + 1
    ni = np.nonzero(derivative > 0)[0] + 1

    # sanity check
    if len(pi) == 0 or len(ni) == 0:
        raise ValueError("NeuroKit error: eda_findpeaks(): Could not find enough",
                         " SCR peaks. Try another method.")

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
    i0 = peaks - (onsets - peaks) / 2.
    if i0[0] < 0:
        i0[0] = 0

    # amplitude
    amplitudes = np.array([np.max(eda_phasic[peaks[i]:onsets[i]]) for i in range(li)])

    # output
    info = {"SCR_Onset": onsets,
            "SCR_Peaks": peaks,
            "SCR_Amplitude": amplitudes}
    return info






def _eda_findpeaks_kim2004(eda_phasic, sampling_rate=1000, amplitude_min=0.1):
    """KBK method to extract Skin Conductivity Responses (SCR) from an
    EDA signal following the approach by Kim et al. (2004).

    Parameters
    ----------
    signal : array
        Input filterd EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    amplitude_min : float, optional
        Minimum treshold by which to exclude SCRs.

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
    - Kim, K. H., Bang, S. W., & Kim, S. R. (2004). Emotion recognition system using short-term monitoring of physiological signals. Medical and biological engineering and computing, 42(3), 419-427.
    """

    # differentiation
    df = np.diff(eda_phasic)

    # smooth
    df = signal_smooth(signal=df, kernel='bartlett', size=int(sampling_rate))

    # zero crosses
    zeros = signal_zerocrossings(df)
    if np.all(df[:zeros[0]] > 0):
        zeros = zeros[1:]
    if np.all(df[zeros[-1]:] > 0):
        zeros = zeros[:-1]

    # exclude SCRs with small amplitude
    thr = amplitude_min * np.max(df)

    scrs, amps, ZC, pks = [], [], [], []
    for i in range(0, len(zeros) - 1, 2):
        scrs += [df[zeros[i]:zeros[i + 1]]]
        aux = scrs[-1].max()
        if aux > thr:
            amps += [aux]
            ZC += [zeros[i]]
            ZC += [zeros[i + 1]]
            pks += [zeros[i] + np.argmax(df[zeros[i]:zeros[i + 1]])]

    scrs = np.array(scrs)
    amps = np.array(amps)
    ZC = np.array(ZC)
    pks = np.array(pks)
    onsets = ZC[::2]

    # output
    info = {"SCR_Onset": onsets,
            "SCR_Peaks": pks,
            "SCR_Amplitude": amps}

    return info
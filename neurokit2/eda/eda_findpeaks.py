# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from ..signal import signal_filter, signal_findpeaks, signal_smooth, signal_zerocrossings


def eda_findpeaks(eda_phasic, sampling_rate=1000, method="neurokit", amplitude_min=0.1):
    """**Find Skin Conductance Responses (SCR) in Electrodermal Activity (EDA)**

    Low-level function used by `eda_peaks()` to identify Skin Conductance Responses (SCR) peaks in
    the phasic component of Electrodermal Activity (EDA) with different possible methods. See
    :func:`eda_peaks` for details.

    Parameters
    ----------
    eda_phasic : Union[list, np.array, pd.Series]
        The phasic component of the EDA signal (from :func:`eda_phasic`).
    sampling_rate : int
        The sampling frequency of the EDA signal (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of ``"neurokit"`` (default),
        ``"gamboa2008"``, ``"kim2004"`` (the default in BioSPPy), ``"vanhalem2020"`` or ``"nabian2018"``.
    amplitude_min : float
        Only used if "method" is ``"neurokit"`` or ``"kim2004"``. Minimum threshold by which to
        exclude SCRs (peaks) as relative to the largest amplitude in the signal.

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the aplitude of the SCR, the
        samples at which the SCR onset and the SCR peaks occur. Accessible with the keys
        ``"SCR_Amplitude"``, ``"SCR_Onsets"``, and ``"SCR_Peaks"`` respectively.

    See Also
    --------
    eda_simulate, eda_clean, eda_phasic, eda_fixpeaks, eda_peaks, eda_process, eda_plot


    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      # Get phasic component
      eda_signal = nk.eda_simulate(duration=30, scr_number=5, drift=0.1, noise=0)
      eda_cleaned = nk.eda_clean(eda_signal)
      eda = nk.eda_phasic(eda_cleaned)
      eda_phasic = eda["EDA_Phasic"].values

      # Find peaks
      gamboa2008 = nk.eda_findpeaks(eda_phasic, method="gamboa2008")
      kim2004 = nk.eda_findpeaks(eda_phasic, method="kim2004")
      neurokit = nk.eda_findpeaks(eda_phasic, method="neurokit")
      vanhalem2020 = nk.eda_findpeaks(eda_phasic, method="vanhalem2020")
      nabian2018 = nk.eda_findpeaks(eda_phasic, method="nabian2018")
      @savefig p_eda_findpeaks.png scale=100%
      nk.events_plot([gamboa2008["SCR_Peaks"], kim2004["SCR_Peaks"], vanhalem2020["SCR_Peaks"],
                           neurokit["SCR_Peaks"], nabian2018["SCR_Peaks"]], eda_phasic)
      @suppress
      plt.close()

    References
    ----------
    * Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci and electrophysiology.
      PhD Thesis Universidade.
    * Kim, K. H., Bang, S. W., & Kim, S. R. (2004). Emotion recognition system using short-term
      monitoring of physiological signals. Medical and biological engineering and computing, 42(3),
      419-427.
    * van Halem, S., Van Roekel, E., Kroencke, L., Kuper, N., & Denissen, J. (2020).
      Moments That Matter? On the Complexity of Using Triggers Based on Skin Conductance to Sample
      Arousing Events Within an Experience Sampling Framework. European Journal of Personality.
    * Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., & Ostadabbas, S. (2018). An
      Open-Source Feature Extraction Tool for the Analysis of Peripheral Physiological Data. IEEE
      journal of translational engineering in health and medicine, 6, 2800711.

    """
    # Try to retrieve the right column if a dataframe is passed
    if isinstance(eda_phasic, pd.DataFrame):
        try:
            eda_phasic = eda_phasic["EDA_Phasic"]
        except KeyError:
            raise KeyError(
                "NeuroKit error: eda_findpeaks(): Please provide an array as the input signal."
            )

    method = method.lower()  # remove capitalised letters
    if method in ["gamboa2008", "gamboa"]:
        info = _eda_findpeaks_gamboa2008(eda_phasic)
    elif method in ["kim", "kbk", "kim2004", "biosppy"]:
        info = _eda_findpeaks_kim2004(
            eda_phasic, sampling_rate=sampling_rate, amplitude_min=amplitude_min
        )
    elif method in ["nk", "nk2", "neurokit", "neurokit2"]:
        info = _eda_findpeaks_neurokit(eda_phasic, amplitude_min=amplitude_min)
    elif method in ["vanhalem2020", "vanhalem", "halem2020"]:
        info = _eda_findpeaks_vanhalem2020(eda_phasic, sampling_rate=sampling_rate)
    elif method in ["nabian2018", "nabian"]:
        info = _eda_findpeaks_nabian2018(eda_phasic)
    else:
        raise ValueError(
            "NeuroKit error: eda_findpeaks(): 'method' should be one of 'neurokit', 'gamboa2008', 'kim2004'"
            " 'vanhalem2020' or 'nabian2018'."
        )

    return info


# =============================================================================
# Methods
# =============================================================================


def _eda_findpeaks_neurokit(eda_phasic, amplitude_min=0.1):
    peaks = signal_findpeaks(eda_phasic, relative_height_min=amplitude_min, relative_max=True)

    info = {
        "SCR_Onsets": peaks["Onsets"],
        "SCR_Peaks": peaks["Peaks"],
        "SCR_Height": eda_phasic[peaks["Peaks"]],
    }

    return info


def _eda_findpeaks_vanhalem2020(eda_phasic, sampling_rate=1000):
    """Follows approach of van Halem et al. (2020).

    A peak is considered when there is a consistent increase of 0.5 seconds following a consistent
    decrease of 0.5 seconds.

    * van Halem, S., Van Roekel, E., Kroencke, L., Kuper, N., & Denissen, J. (2020).
      Moments That Matter? On the Complexity of Using Triggers Based on Skin Conductance to Sample
      Arousing Events Within an Experience Sampling Framework. European Journal of Personality.

    """
    # smooth
    eda_phasic = signal_filter(
        eda_phasic,
        sampling_rate=sampling_rate,
        lowcut=None,
        highcut=None,
        method="savgol",
        window_size=501,
    )
    info = signal_findpeaks(eda_phasic)
    peaks = info["Peaks"]

    threshold = 0.5 * sampling_rate

    # Define each peak as a consistent increase of 0.5s
    increase = info["Peaks"] - info["Onsets"]
    peaks = peaks[increase > threshold]
    idx = np.where(peaks[:, None] == info["Peaks"][None, :])[1]

    # Check if each peak is followed by consistent decrease of 0.5s
    decrease = info["Offsets"][idx] - peaks
    if any(np.isnan(decrease)):
        decrease[np.isnan(decrease)] = False
    if any(decrease < threshold):
        keep = np.where(decrease > threshold)[0]
        idx = idx[keep]  # Update index

    info = {
        "SCR_Onsets": info["Onsets"][idx],
        "SCR_Peaks": info["Peaks"][idx],
        "SCR_Height": eda_phasic[info["Peaks"][idx]],
    }

    return info


def _eda_findpeaks_gamboa2008(eda_phasic):
    """Basic method to extract Skin Conductivity Responses (SCR) from an EDA signal following the
    approach in the thesis by Gamboa (2008).

    * Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci and electrophysiology.
      PhD Thesis Universidade.

    """
    derivative = np.diff(np.sign(np.diff(eda_phasic)))

    # find extrema
    pi = np.nonzero(derivative < 0)[0] + 1
    ni = np.nonzero(derivative > 0)[0] + 1

    # sanity check
    if len(pi) == 0 or len(ni) == 0:
        raise ValueError(
            "NeuroKit error: eda_findpeaks(): Could not find enough SCR peaks. Try another method."
        )

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

    * Kim, K. H., Bang, S. W., & Kim, S. R. (2004). Emotion recognition system using short-term
      monitoring of physiological signals. Medical and biological engineering and computing, 42(3),
      419-427.

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

    scrs, amps, ZC, pks = [], [], [], []
    for i in range(0, len(zeros) - 1, 2):
        scrs += [eda_phasic[zeros[i] : zeros[i + 1]]]
        aux = scrs[-1].max()
        if aux > 0:
            amps += [aux]
            ZC += [zeros[i]]
            ZC += [zeros[i + 1]]
            pks += [zeros[i] + np.argmax(eda_phasic[zeros[i] : zeros[i + 1]])]

    amps = np.array(amps)
    ZC = np.array(ZC)
    pks = np.array(pks)
    onsets = ZC[::2]

    # exclude SCRs with small amplitude
    masked = amps > (amplitude_min * np.nanmax(amps))  # threshold
    amps = amps[masked]
    pks = pks[masked]
    onsets = onsets[masked]

    # output
    info = {"SCR_Onsets": onsets, "SCR_Peaks": pks, "SCR_Height": amps}

    return info


def _eda_findpeaks_nabian2018(eda_phasic):
    """Basic method to extract Skin Conductivity Responses (SCR) from an EDA signal following the
    approach by Nabian et al. (2018). The amplitude of the SCR is obtained by finding the maximum
    value between these two zero-crossings, and calculating the difference between the initial zero
    crossing and the maximum value. Detected SCRs with amplitudes smaller than 10 percent of the
    maximum SCR amplitudes that are already detected on the differentiated signal will be
    eliminated. It is crucial that artifacts are removed before finding peaks.

    * Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., & Ostadabbas, S. (2018). An
      Open-Source Feature Extraction Tool for the Analysis of Peripheral Physiological Data. IEEE
      journal of translational engineering in health and medicine, 6, 2800711.
      https://doi.org/10.1109/JTEHM.2018.2878000

    """

    # differentiation
    eda_phasic_diff = np.diff(eda_phasic)

    # smooth
    eda_phasic_smoothed = signal_smooth(eda_phasic_diff, kernel="bartlett", size=20)

    # zero crossings
    pos_crossings = signal_zerocrossings(eda_phasic_smoothed, direction="positive")
    neg_crossings = signal_zerocrossings(eda_phasic_smoothed, direction="negative")

    # if negative crossing happens before the positive crossing
    # delete first negative crossing because we want to identify peaks
    if neg_crossings[0] < pos_crossings[0]:
        neg_crossings = neg_crossings[1:]
    # Sanitize consecutive crossings

    if len(pos_crossings) > len(neg_crossings):
        pos_crossings = pos_crossings[0 : len(neg_crossings)]
    elif len(pos_crossings) < len(neg_crossings):
        neg_crossings = neg_crossings[0 : len(pos_crossings)]

    peaks_list = []
    onsets_list = []
    amps_list = []
    for i, j in zip(pos_crossings, neg_crossings):
        window = eda_phasic[i:j]
        # The amplitude of the SCR is obtained by finding the maximum value
        # between these two zero-crossings and calculating the difference
        # between the initial zero crossing and the maximum value.
        # amplitude defined in neurokit2
        amp = np.nanmax(window)

        # Detected SCRs with amplitudes less than 10% of max SCR amplitude will be eliminated
        # we append the first SCR
        if len(amps_list) == 0:
            # be careful, if two peaks have the same amplitude, np.where will return a list
            peaks = np.where(eda_phasic == amp)[0]
            # make sure that the peak is within the window
            peaks = [peak for peak in [peaks] if peak > i and peak < j]
            peaks_list.append(peaks[0])
            onsets_list.append(i)
            amps_list.append(amp)
        else:
            # we have a list of peaks
            # amplitude defined in the paper
            diff = amp - eda_phasic[i]
            if not diff < (0.1 * max(amps_list)):
                peaks = np.where(eda_phasic == amp)[0]
                # make sure that the peak is within the window
                peaks = [peak for peak in [peaks] if peak > i and peak < j]
                peaks_list.append(peaks[0])
                onsets_list.append(i)
                amps_list.append(amp)

    # output
    info = {
        "SCR_Onsets": np.array(onsets_list),
        "SCR_Peaks": np.hstack(np.array(peaks_list)),
        "SCR_Height": np.array(amps_list),
    }

    return info

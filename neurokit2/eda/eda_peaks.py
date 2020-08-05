# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from ..misc import find_closest
from ..signal import signal_formatpeaks
from .eda_findpeaks import eda_findpeaks
from .eda_fixpeaks import eda_fixpeaks


def eda_peaks(eda_phasic, sampling_rate=1000, method="neurokit", amplitude_min=0.1):
    """Identify Skin Conductance Responses (SCR) in Electrodermal Activity (EDA).

    Identify Skin Conductance Responses (SCR) peaks in the phasic component of
    Electrodermal Activity (EDA) with different possible methods, such as:

    - `Gamboa, H. (2008)
    <http://www.lx.it.pt/~afred/pub/thesisHugoGamboa.pdf>`_
    - `Kim et al. (2004)
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.102.7385&rep=rep1&type=pdf>`_

    Parameters
    ----------
    eda_phasic : Union[list, np.array, pd.Series]
        The phasic component of the EDA signal (from `eda_phasic()`).
    sampling_rate : int
        The sampling frequency of the EDA signal (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of "neurokit" (default),
        "gamboa2008", "kim2004" (the default in BioSPPy), "vanhalem2020" or "nabian2018".
    amplitude_min : float
        Only used if 'method' is 'neurokit' or 'kim2004'. Minimum threshold by which to exclude
        SCRs (peaks) as relative to the largest amplitude in the signal.

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the aplitude of the SCR, the samples
        at which the SCR onset and the SCR peaks occur. Accessible with the keys "SCR_Amplitude",
        "SCR_Onsets", and "SCR_Peaks" respectively.
    signals : DataFrame
        A DataFrame of same length as the input signal in which occurences of SCR peaks are marked as
        "1" in lists of zeros with the same length as `eda_cleaned`. Accessible with the keys "SCR_Peaks".

    See Also
    --------
    eda_simulate, eda_clean, eda_phasic, eda_process, eda_plot



    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> # Get phasic component
    >>> eda_signal = nk.eda_simulate(duration=30, scr_number=5, drift=0.1, noise=0, sampling_rate=100)
    >>> eda_cleaned = nk.eda_clean(eda_signal, sampling_rate=100)
    >>> eda = nk.eda_phasic(eda_cleaned, sampling_rate=100)
    >>> eda_phasic = eda["EDA_Phasic"].values
    >>>
    >>> # Find peaks
    >>> _, kim2004 = nk.eda_peaks(eda_phasic, method="kim2004")
    >>> _, neurokit = nk.eda_peaks(eda_phasic, method="neurokit")
    >>> _, nabian2018 = nk.eda_peaks(eda_phasic, method="nabian2018")
    >>> nk.events_plot([nabian2018["SCR_Peaks"], kim2004["SCR_Peaks"], neurokit["SCR_Peaks"]], eda_phasic) #doctest: +ELLIPSIS
    <Figure ...>

    References
    ----------
    - Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci and electrophysiology.
      PhD ThesisUniversidade.
    - Kim, K. H., Bang, S. W., & Kim, S. R. (2004). Emotion recognition system using short-term monitoring
      of physiological signals. Medical and biological engineering and computing, 42(3), 419-427.
    - van Halem, S., Van Roekel, E., Kroencke, L., Kuper, N., & Denissen, J. (2020).
      Moments That Matter? On the Complexity of Using Triggers Based on Skin Conductance to Sample
      Arousing Events Within an Experience Sampling Framework. European Journal of Personality.
    - Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., & Ostadabbas, S. (2018). An
      Open-Source Feature Extraction Tool for the Analysis of Peripheral Physiological Data. IEEE
      journal of translational engineering in health and medicine, 6, 2800711.
      https://doi.org/10.1109/JTEHM.2018.2878000

    """
    if isinstance(eda_phasic, (pd.DataFrame, pd.Series)):
        try:
            eda_phasic = eda_phasic["EDA_Phasic"]
        except KeyError:
            eda_phasic = eda_phasic.values

    # Get basic
    info = eda_findpeaks(eda_phasic, sampling_rate=sampling_rate, method=method, amplitude_min=amplitude_min)
    info = eda_fixpeaks(info)

    # Get additional features (rise time, half recovery time, etc.)
    info = _eda_peaks_getfeatures(info, eda_phasic, sampling_rate, recovery_percentage=0.5)

    # Prepare output.
    peak_signal = signal_formatpeaks(info, desired_length=len(eda_phasic), peak_indices=info["SCR_Peaks"])

    return peak_signal, info


# =============================================================================
# Utility
# =============================================================================


def _eda_peaks_getfeatures(info, eda_phasic, sampling_rate=1000, recovery_percentage=0.5):

    # Sanity checks -----------------------------------------------------------

    # Peaks (remove peaks with no onset)
    valid_peaks = np.logical_and(
        info["SCR_Peaks"] > np.nanmin(info["SCR_Onsets"]), ~np.isnan(info["SCR_Onsets"])
    )  # pylint: disable=E1111
    peaks = info["SCR_Peaks"][valid_peaks]

    # Onsets (remove onsets with no peaks)
    valid_onsets = ~np.isnan(info["SCR_Onsets"])
    valid_onsets[valid_onsets] = info["SCR_Onsets"][valid_onsets] < np.nanmax(info["SCR_Peaks"])
    onsets = info["SCR_Onsets"][valid_onsets].astype(np.int)

    if len(onsets) != len(peaks):
        raise ValueError(
            "NeuroKit error: eda_peaks(): Peaks and onsets don't ",
            "match, so cannot get amplitude safely. Check why using `find_peaks()`.",
        )

    # Amplitude and Rise Time -------------------------------------------------

    # Amplitudes
    amplitude = np.full(len(info["SCR_Height"]), np.nan)
    amplitude[valid_peaks] = info["SCR_Height"][valid_peaks] - eda_phasic[onsets]

    # Rise times
    risetime = np.full(len(info["SCR_Peaks"]), np.nan)
    risetime[valid_peaks] = (peaks - onsets) / sampling_rate

    # Save info
    info["SCR_Amplitude"] = amplitude
    info["SCR_RiseTime"] = risetime

    # Recovery time -----------------------------------------------------------

    # (Half) Recovery times
    recovery = np.full(len(info["SCR_Peaks"]), np.nan)
    recovery_time = np.full(len(info["SCR_Peaks"]), np.nan)
    recovery_values = eda_phasic[onsets] + (amplitude[valid_peaks] * recovery_percentage)

    for i, peak_index in enumerate(peaks):
        # Get segment between peak and next peak
        try:
            segment = eda_phasic[peak_index : peaks[i + 1]]
        except IndexError:
            segment = eda_phasic[peak_index::]

        # Adjust segment (cut when it reaches minimum to avoid picking out values on the rise of the next peak)
        segment = segment[0 : np.argmin(segment)]

        # Find recovery time
        recovery_value = find_closest(recovery_values[i], segment, direction="smaller", strictly=False)

        # Detect recovery points only if there are datapoints below recovery value
        if np.min(segment) < recovery_value:
            segment_index = np.where(segment == recovery_value)[0][0]
            recovery[np.where(valid_peaks)[0][i]] = peak_index + segment_index
            recovery_time[np.where(valid_peaks)[0][i]] = segment_index / sampling_rate

    # Save ouput
    info["SCR_Recovery"] = recovery
    info["SCR_RecoveryTime"] = recovery_time

    return info

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import pywt

from ..epochs import epochs_create, epochs_to_df
from ..signal import (
    signal_findpeaks,
    signal_formatpeaks,
    signal_rate,
    signal_resample,
    signal_smooth,
    signal_zerocrossings,
)
from ..stats import standardize
from .ecg_peaks import ecg_peaks
from .ecg_segment import ecg_segment


def ecg_delineate(
    ecg_cleaned,
    rpeaks=None,
    sampling_rate=1000,
    method="dwt",
    show=False,
    show_type="peaks",
    check=False,
    **kwargs
):
    """**Delineate QRS complex**

    Function to delineate the QRS complex, i.e., the different waves of the cardiac cycles. A
    typical ECG heartbeat consists of a P wave, a QRS complex and a T wave. The P wave represents
    the wave of depolarization that spreads from the SA-node throughout the atria. The QRS complex
    reflects the rapid depolarization of the right and left ventricles. Since the ventricles are
    the largest part of the heart, in terms of mass, the QRS complex usually has a much larger
    amplitude than the P-wave. The T wave represents the ventricular repolarization of the
    ventricles.On rare occasions, a U wave can be seen following the T wave. The U wave is believed
    to be related to the last remnants of ventricular repolarization.

    Parameters
    ----------
    ecg_cleaned : Union[list, np.array, pd.Series]
        The cleaned ECG channel as returned by ``ecg_clean()``.
    rpeaks : Union[list, np.array, pd.Series]
        The samples at which R-peaks occur. Accessible with the key "ECG_R_Peaks" in the info
        dictionary returned by ``ecg_findpeaks()``.
    sampling_rate : int
        The sampling frequency of ``ecg_signal`` (in Hz, i.e., samples/second). Defaults to 1000.
    method : str
        Can be one of ``"peak"`` for a peak-based method, ``"prominence"`` for a peak-prominence-based
        method (Emrich et al., 2024), ``"cwt"`` for continuous wavelet transform or ``"dwt"`` (default)
        for discrete wavelet transform.
        The ``"prominence"`` method might be useful to detect the waves, allowing to set individual physiological
        limits (see kwargs), while the ``"dwt"`` method might be more precise for detecting the onsets and offsets
        of the waves (but might exhibit lower accuracy when there is significant variation in wave morphology).
        The ``"peak"`` method, which uses the zero-crossings of the signal derivatives, works best with very clean signals.
    show : bool
        If ``True``, will return a plot to visualizing the delineated waves information.
    show_type: str
        The type of delineated waves information showed in the plot.
        Can be ``"peaks"``, ``"bounds_R"``, ``"bounds_T"``, ``"bounds_P"`` or ``"all"``.
    check : bool
        Defaults to ``False``. If ``True``, replaces the delineated features with ``np.nan`` if its
        standardized distance from R-peaks is more than 3.
    **kwargs
        Other optional arguments:
        If using the ``"prominence"`` method, additional parameters (in milliseconds) can be passed to set
        individual physiological limits for the search boundaries:
        - ``max_qrs_interval``: The maximum allowable QRS complex interval. Defaults to 180 ms.
        - ``max_pr_interval``: The maximum PR interval duration. Defaults to 300 ms.
        - ``max_r_rise_time``: Maximum duration for the R-wave rise. Defaults to 120 ms.
        - ``typical_st_segment``: Typical duration of the ST segment. Defaults to 150 ms.
        - ``max_p_basepoint_interval``: The maximum interval between P-wave on- and offset. Defaults to 100 ms.
        - ``max_r_basepoint_interval``: The maximum interval between R-wave on- and offset. Defaults to 100 ms.
        - ``max_t_basepoint_interval``: The maximum interval between T-wave on- and offset. Defaults to 200 ms.

    Returns
    -------
    waves : dict
        A dictionary containing additional information.
        For derivative method, the dictionary contains the samples at which P-peaks, Q-peaks,
        S-peaks, T-peaks, P-onsets and T-offsets occur, accessible with the keys ``"ECG_P_Peaks"``,
        ``"ECG_Q_Peaks"``, ``"ECG_S_Peaks"``, ``"ECG_T_Peaks"``, ``"ECG_P_Onsets"``,
        ``"ECG_T_Offsets"``, respectively.

        For the wavelet and prominence methods, in addition to the above information, the dictionary contains the
        samples at which QRS-onsets and QRS-offsets occur, accessible with the key
        ``"ECG_P_Peaks"``, ``"ECG_T_Peaks"``, ``"ECG_P_Onsets"``, ``"ECG_P_Offsets"``,
        ``"ECG_Q_Peaks"``, ``"ECG_S_Peaks"``, ``"ECG_T_Onsets"``, ``"ECG_T_Offsets"``,
        ``"ECG_R_Onsets"``, ``"ECG_R_Offsets"``, respectively.

    signals : DataFrame
        A DataFrame of same length as the input signal in which occurrences of
        peaks, onsets and offsets marked as "1" in a list of zeros.

    See Also
    --------
    ecg_clean, .signal_fixpeaks, ecg_peaks, .signal_rate, ecg_process, ecg_plot

    Examples
    --------
    * Step 1. Delineate

    .. ipython:: python

      import neurokit2 as nk

      # Simulate ECG signal
      ecg = nk.ecg_simulate(duration=10, sampling_rate=1000)
      # Get R-peaks location
      _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=1000)
      # Delineate cardiac cycle
      signals, waves = nk.ecg_delineate(ecg, rpeaks, sampling_rate=1000)

    * Step 2. Plot P-Peaks and T-Peaks

    .. ipython:: python

      @savefig p_ecg_delineate1.png scale=100%
      nk.events_plot([waves["ECG_P_Peaks"], waves["ECG_T_Peaks"]], ecg)
      @suppress
      plt.close()

    References
    --------------
    - Martínez, J. P., Almeida, R., Olmos, S., Rocha, A. P., & Laguna, P. (2004). A wavelet-based
      ECG delineator: evaluation on standard databases. IEEE Transactions on biomedical engineering,
      51(4), 570-581.
    - Emrich, J., Gargano, A., Koka, T., & Muma, M. (2024). Physiology-Informed ECG Delineation Based
      on Peak Prominence. 32nd European Signal Processing Conference (EUSIPCO), 1402-1406.

    """
    # Sanitize input for ecg_cleaned
    if isinstance(ecg_cleaned, pd.DataFrame):
        cols = [col for col in ecg_cleaned.columns if "ECG_Clean" in col]
        if cols:
            ecg_cleaned = ecg_cleaned[cols[0]].values
        else:
            raise ValueError(
                "NeuroKit error: ecg_delineate(): Wrong input, we couldn't extract"
                "cleaned signal."
            )

    elif isinstance(ecg_cleaned, dict):
        for i in ecg_cleaned:
            cols = [col for col in ecg_cleaned[i].columns if "ECG_Clean" in col]
            if cols:
                signals = epochs_to_df(ecg_cleaned)
                ecg_cleaned = signals[cols[0]].values

            else:
                raise ValueError(
                    "NeuroKit error: ecg_delineate(): Wrong input, we couldn't extract"
                    "cleaned signal."
                )

    elif isinstance(ecg_cleaned, pd.Series):
        ecg_cleaned = ecg_cleaned.values

    # Sanitize input for rpeaks
    if rpeaks is None:
        _, rpeaks = ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        rpeaks = rpeaks["ECG_R_Peaks"]

    if isinstance(rpeaks, dict):
        rpeaks = rpeaks["ECG_R_Peaks"]

    method = method.lower()  # remove capitalised letters
    if method in ["peak", "peaks", "derivative", "gradient"]:
        waves = _ecg_delineator_peak(
            ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate
        )
    elif method in ["cwt", "continuous wavelet transform"]:
        waves = _ecg_delineator_cwt(
            ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate
        )
    elif method in ["dwt", "discrete wavelet transform"]:
        waves = _dwt_ecg_delineator(ecg_cleaned, rpeaks, sampling_rate=sampling_rate)
    elif method in ["prominence", "peak-prominence", "emrich", "emrich2024"]:
        waves = _prominence_ecg_delineator(ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate, **kwargs)

    else:
        raise ValueError(
            "NeuroKit error: ecg_delineate(): 'method' should be one of 'peak', 'prominence',"
            "'cwt' or 'dwt'."
        )

    # Ensure that all indices are not larger than ECG signal indices
    for _, value in waves.items():
        if value[-1] >= len(ecg_cleaned):
            value[-1] = np.nan

    # Remove NaN in Peaks, Onsets, and Offsets
    waves_noNA = waves.copy()
    for feature in waves_noNA.keys():
        waves_noNA[feature] = [
            int(x) for x in waves_noNA[feature] if ~np.isnan(x) and x > 0
        ]

    instant_peaks = signal_formatpeaks(waves_noNA, desired_length=len(ecg_cleaned))
    signals = instant_peaks

    waves_sanitized = {}
    for feature, values in waves.items():
        waves_sanitized[feature] = [x for x in values if x > 0 or x is np.nan]

    if show is True:
        _ecg_delineate_plot(
            ecg_cleaned,
            rpeaks=rpeaks,
            signals=signals,
            signal_features_type=show_type,
            sampling_rate=sampling_rate,
            **kwargs
        )

    if check is True:
        waves_sanitized = _ecg_delineate_check(waves_sanitized, rpeaks)

    return signals, waves_sanitized


# =============================================================================
# WAVELET METHOD (DWT)
# =============================================================================
def _dwt_resample_points(peaks, sampling_rate, desired_sampling_rate):
    """Resample given points to a different sampling rate."""
    if isinstance(
        peaks, np.ndarray
    ):  # peaks are passed in from previous processing steps
        # Prevent overflow by converting to np.int64 (peaks might be passed in containing np.int32).
        peaks = peaks.astype(dtype=np.int64)
    elif isinstance(peaks, list):  # peaks returned from internal functions
        # Cannot be converted to int since list might contain np.nan. Automatically cast to np.float64 if list contains np.nan.
        peaks = np.array(peaks)
    peaks_resample = peaks * desired_sampling_rate / sampling_rate
    peaks_resample = [
        np.nan if np.isnan(x) else int(x) for x in peaks_resample.tolist()
    ]
    return peaks_resample


def _dwt_ecg_delineator(ecg, rpeaks, sampling_rate, analysis_sampling_rate=2000):
    """Delinate ecg signal using discrete wavelet transforms.

    Parameters
    ----------
    ecg : Union[list, np.array, pd.Series]
        The cleaned ECG channel as returned by `ecg_clean()`.
    rpeaks : Union[list, np.array, pd.Series]
        The samples at which R-peaks occur. Accessible with the key "ECG_R_Peaks" in the info dictionary
        returned by `ecg_findpeaks()`.
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
    analysis_sampling_rate : int
        The sampling frequency for analysis (in Hz, i.e., samples/second).

    Returns
    --------
    dict
        Dictionary of the points.

    """
    # No dwt defined method for Q and S peak
    # Adopting manual method from "peak" method
    qpeaks = []
    speaks = []
    heartbeats = ecg_segment(ecg, rpeaks, sampling_rate=sampling_rate)
    for i, rpeak in enumerate(rpeaks):
        heartbeat = heartbeats[str(i + 1)]
        # Get index of R peaks
        R = heartbeat.index.get_loc(
            np.min(heartbeat.index.values[heartbeat.index.values > 0])
        )
        # Q wave
        Q_index, Q = _ecg_delineator_peak_Q(rpeak, heartbeat, R)
        qpeaks.append(Q_index)
        # S wave
        S_index, S = _ecg_delineator_peak_S(rpeak, heartbeat)
        speaks.append(S_index)

    # dwt to delineate tp waves, onsets, offsets and qrs ontsets and offsets
    ecg = signal_resample(
        ecg, sampling_rate=sampling_rate, desired_sampling_rate=analysis_sampling_rate
    )
    dwtmatr = _dwt_compute_multiscales(ecg, 9)

    # # only for debugging
    # for idx in [0, 1, 2, 3]:
    #     plt.plot(dwtmatr[idx + 3], label=f'W[{idx}]')
    # plt.plot(ecg, '--')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    rpeaks_resampled = _dwt_resample_points(
        rpeaks, sampling_rate, analysis_sampling_rate
    )
    qpeaks_resampled = _dwt_resample_points(
        qpeaks, sampling_rate, analysis_sampling_rate
    )

    tpeaks, ppeaks = _dwt_delineate_tp_peaks(
        ecg, rpeaks_resampled, dwtmatr, sampling_rate=analysis_sampling_rate
    )
    qrs_onsets, qrs_offsets = _dwt_delineate_qrs_bounds(
        rpeaks_resampled,
        dwtmatr,
        ppeaks,
        tpeaks,
        qpeaks_resampled,
        sampling_rate=analysis_sampling_rate,
    )
    ponsets, poffsets = _dwt_delineate_tp_onsets_offsets(
        ppeaks, rpeaks_resampled, dwtmatr, sampling_rate=analysis_sampling_rate
    )
    tonsets, toffsets = _dwt_delineate_tp_onsets_offsets(
        tpeaks,
        rpeaks_resampled,
        dwtmatr,
        sampling_rate=analysis_sampling_rate,
        onset_weight=0.6,
        duration_onset=0.6,
    )

    return dict(
        ECG_P_Peaks=_dwt_resample_points(
            ppeaks, analysis_sampling_rate, desired_sampling_rate=sampling_rate
        ),
        ECG_P_Onsets=_dwt_resample_points(
            ponsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate
        ),
        ECG_P_Offsets=_dwt_resample_points(
            poffsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate
        ),
        ECG_Q_Peaks=qpeaks,
        ECG_R_Onsets=_dwt_resample_points(
            qrs_onsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate
        ),
        ECG_R_Offsets=_dwt_resample_points(
            qrs_offsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate
        ),
        ECG_S_Peaks=speaks,
        ECG_T_Peaks=_dwt_resample_points(
            tpeaks, analysis_sampling_rate, desired_sampling_rate=sampling_rate
        ),
        ECG_T_Onsets=_dwt_resample_points(
            tonsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate
        ),
        ECG_T_Offsets=_dwt_resample_points(
            toffsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate
        ),
    )


def _dwt_adjust_parameters(rpeaks, sampling_rate, duration=None, target=None):
    average_rate = np.median(signal_rate(peaks=rpeaks, sampling_rate=sampling_rate))
    if target == "degree":
        # adjust defree of dwt by sampling_rate and HR
        scale_factor = (sampling_rate / 250) / (average_rate / 60)
        return int(np.log2(scale_factor))
    elif target == "duration":
        # adjust duration of search by HR
        return np.round(duration * (60 / average_rate), 3)


def _dwt_delineate_tp_peaks(
    ecg,
    rpeaks,
    dwtmatr,
    sampling_rate=250,
    qrs_width=0.13,
    p2r_duration=0.2,
    rt_duration=0.25,
    degree_tpeak=3,
    degree_ppeak=2,
    epsilon_T_weight=0.25,
    epsilon_P_weight=0.02,
):
    """
    Parameters
    ----------
    ecg : Union[list, np.array, pd.Series]
        The cleaned ECG channel as returned by `ecg_clean()`.
    rpeaks : Union[list, np.array, pd.Series]
        The samples at which R-peaks occur. Accessible with the key "ECG_R_Peaks" in the info dictionary
        returned by `ecg_findpeaks()`.
    dwtmatr : np.array
        Output of `_dwt_compute_multiscales()`. Multiscales of wavelet transform.
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
    qrs_width : int
        Approximate duration of qrs in seconds. Default to 0.13 seconds.
    p2r_duration : int
        Approximate duration from P peaks to R peaks in seconds.
    rt_duration : int
        Approximate duration from R peaks to T peaks in secons.
    degree_tpeak : int
        Wavelet transform of scales 2**3.
    degree_tpeak : int
        Wavelet transform of scales 2**2.
    epsilon_T_weight : int
        Epsilon of RMS value of wavelet transform. Appendix (A.3).
    epsilon_P_weight : int
        Epsilon of RMS value of wavelet transform. Appendix (A.4).
    """
    srch_bndry = int(0.5 * qrs_width * sampling_rate)
    degree_add = _dwt_adjust_parameters(rpeaks, sampling_rate, target="degree")
    # sanitize search duration by HR
    p2r_duration = _dwt_adjust_parameters(
        rpeaks, sampling_rate, duration=p2r_duration, target="duration"
    )
    rt_duration = _dwt_adjust_parameters(
        rpeaks, sampling_rate, duration=rt_duration, target="duration"
    )

    tpeaks = []
    for rpeak_ in rpeaks:
        if np.isnan(rpeak_):
            tpeaks.append(np.nan)
            continue
        # search for T peaks from R peaks
        srch_idx_start = rpeak_ + srch_bndry
        srch_idx_end = rpeak_ + 2 * int(rt_duration * sampling_rate)
        dwt_local = dwtmatr[degree_tpeak + degree_add, srch_idx_start:srch_idx_end]
        if len(dwt_local) == 0:
            tpeaks.append(np.nan)
            continue

        height = epsilon_T_weight * np.sqrt(np.mean(np.square(dwt_local)))
        ecg_local = ecg[srch_idx_start:srch_idx_end]
        peaks, __ = scipy.signal.find_peaks(np.abs(dwt_local), height=height)
        peaks = list(
            filter(lambda p: np.abs(dwt_local[p]) > 0.025 * max(dwt_local), peaks)
        )  # pylint: disable=W0640
        if dwt_local[0] > 0:  # just append
            peaks = [0] + peaks

        # detect morphology
        candidate_peaks = []
        candidate_peaks_scores = []
        for idx_peak, idx_peak_nxt in zip(peaks[:-1], peaks[1:]):
            correct_sign = (
                dwt_local[idx_peak] > 0 and dwt_local[idx_peak_nxt] < 0
            )  # pylint: disable=R1716
            if correct_sign:
                idx_zero = (
                    signal_zerocrossings(dwt_local[idx_peak : idx_peak_nxt + 1])[0]
                    + idx_peak
                )
                # This is the score assigned to each peak. The peak with the highest score will be
                # selected.
                score = ecg_local[idx_zero] - (
                    float(idx_zero) / sampling_rate - (rt_duration - 0.5 * qrs_width)
                )
                candidate_peaks.append(idx_zero)
                candidate_peaks_scores.append(score)

        if not candidate_peaks:
            tpeaks.append(np.nan)
            continue

        tpeaks.append(
            candidate_peaks[np.argmax(candidate_peaks_scores)] + srch_idx_start
        )

    ppeaks = []
    for rpeak in rpeaks:
        if np.isnan(rpeak):
            ppeaks.append(np.nan)
            continue

        # search for P peaks from Rpeaks
        srch_idx_start = rpeak - 2 * int(p2r_duration * sampling_rate)
        srch_idx_end = rpeak - srch_bndry
        dwt_local = dwtmatr[degree_ppeak + degree_add, srch_idx_start:srch_idx_end]
        if len(dwt_local) == 0:
            ppeaks.append(np.nan)
            continue

        height = epsilon_P_weight * np.sqrt(np.mean(np.square(dwt_local)))
        ecg_local = ecg[srch_idx_start:srch_idx_end]
        peaks, __ = scipy.signal.find_peaks(np.abs(dwt_local), height=height)
        peaks = list(
            filter(lambda p: np.abs(dwt_local[p]) > 0.025 * max(dwt_local), peaks)
        )
        if dwt_local[0] > 0:  # just append
            peaks = [0] + peaks

        # detect morphology
        candidate_peaks = []
        candidate_peaks_scores = []
        for idx_peak, idx_peak_nxt in zip(peaks[:-1], peaks[1:]):
            correct_sign = (
                dwt_local[idx_peak] > 0 and dwt_local[idx_peak_nxt] < 0
            )  # pylint: disable=R1716
            if correct_sign:
                idx_zero = (
                    signal_zerocrossings(dwt_local[idx_peak : idx_peak_nxt + 1])[0]
                    + idx_peak
                )
                # This is the score assigned to each peak. The peak with the highest score will be
                # selected.
                score = ecg_local[idx_zero] - abs(
                    float(idx_zero) / sampling_rate - p2r_duration
                )  # Minus p2r because of the srch_idx_start
                candidate_peaks.append(idx_zero)
                candidate_peaks_scores.append(score)

        if not candidate_peaks:
            ppeaks.append(np.nan)
            continue

        ppeaks.append(
            candidate_peaks[np.argmax(candidate_peaks_scores)] + srch_idx_start
        )

    return tpeaks, ppeaks


def _dwt_delineate_tp_onsets_offsets(
    peaks,
    rpeaks,
    dwtmatr,
    sampling_rate=250,
    duration_onset=0.3,
    duration_offset=0.3,
    onset_weight=0.4,
    offset_weight=0.4,
    degree_onset=2,
    degree_offset=2,
):
    # sanitize search duration by HR
    duration_onset = _dwt_adjust_parameters(
        rpeaks, sampling_rate, duration=duration_onset, target="duration"
    )
    duration_offset = _dwt_adjust_parameters(
        rpeaks, sampling_rate, duration=duration_offset, target="duration"
    )
    degree = _dwt_adjust_parameters(rpeaks, sampling_rate, target="degree")
    onsets = []
    offsets = []
    for i in range(len(peaks)):  # pylint: disable=C0200
        # look for onsets
        srch_idx_start = peaks[i] - int(duration_onset * sampling_rate)
        srch_idx_end = peaks[i]
        if srch_idx_start is np.nan or srch_idx_end is np.nan:
            onsets.append(np.nan)
            continue
        dwt_local = dwtmatr[degree_onset + degree, srch_idx_start:srch_idx_end]
        onset_slope_peaks, __ = scipy.signal.find_peaks(dwt_local)
        if len(onset_slope_peaks) == 0:
            onsets.append(np.nan)
            continue
        epsilon_onset = onset_weight * dwt_local[onset_slope_peaks[-1]]
        if not (dwt_local[: onset_slope_peaks[-1]] < epsilon_onset).any():
            onsets.append(np.nan)
            continue
        candidate_onsets = np.where(dwt_local[: onset_slope_peaks[-1]] < epsilon_onset)[
            0
        ]
        onsets.append(candidate_onsets[-1] + srch_idx_start)

        # # only for debugging
        # events_plot([candidate_onsets, onset_slope_peaks], dwt_local)
        # plt.plot(ecg[srch_idx_start: srch_idx_end], '--', label='ecg')
        # plt.show()

    for i in range(len(peaks)):  # pylint: disable=C0200
        # look for offset
        srch_idx_start = peaks[i]
        srch_idx_end = peaks[i] + int(duration_offset * sampling_rate)
        if srch_idx_start is np.nan or srch_idx_end is np.nan:
            offsets.append(np.nan)
            continue
        dwt_local = dwtmatr[degree_offset + degree, srch_idx_start:srch_idx_end]
        offset_slope_peaks, __ = scipy.signal.find_peaks(-dwt_local)
        if len(offset_slope_peaks) == 0:
            offsets.append(np.nan)
            continue
        epsilon_offset = -offset_weight * dwt_local[offset_slope_peaks[0]]
        if not (-dwt_local[offset_slope_peaks[0] :] < epsilon_offset).any():
            offsets.append(np.nan)
            continue
        candidate_offsets = (
            np.where(-dwt_local[offset_slope_peaks[0] :] < epsilon_offset)[0]
            + offset_slope_peaks[0]
        )
        offsets.append(candidate_offsets[0] + srch_idx_start)

        # # only for debugging
        # events_plot([candidate_offsets, offset_slope_peaks], dwt_local)
        # plt.plot(ecg[srch_idx_start: srch_idx_end], '--', label='ecg')
        # plt.show()

    return onsets, offsets


def _dwt_delineate_qrs_bounds(
    rpeaks, dwtmatr, ppeaks, tpeaks, qpeaks, sampling_rate=250
):
    degree = _dwt_adjust_parameters(rpeaks, sampling_rate, target="degree")
    onsets = []
    for i in range(len(qpeaks)):  # pylint: disable=C0200
        # look for onsets
        srch_idx_start = ppeaks[i]
        srch_idx_end = qpeaks[i]
        if srch_idx_start is np.nan or srch_idx_end is np.nan:
            onsets.append(np.nan)
            continue
        dwt_local = dwtmatr[2 + degree, srch_idx_start:srch_idx_end]
        onset_slope_peaks, __ = scipy.signal.find_peaks(-dwt_local)
        if len(onset_slope_peaks) == 0:
            onsets.append(np.nan)
            continue
        epsilon_onset = 0.5 * -dwt_local[onset_slope_peaks[-1]]
        if not (-dwt_local[: onset_slope_peaks[-1]] < epsilon_onset).any():
            onsets.append(np.nan)
            continue
        candidate_onsets = np.where(
            -dwt_local[: onset_slope_peaks[-1]] < epsilon_onset
        )[0]
        onsets.append(candidate_onsets[-1] + srch_idx_start)

        # only for debugging
    #        import neurokit as nk
    #        events_plot(candidate_onsets, -dwt_local)
    #        plt.plot(ecg[srch_idx_start: srch_idx_end], '--', label='ecg')
    #        plt.legend()
    #        plt.show()

    offsets = []
    for i in range(len(rpeaks)):  # pylint: disable=C0200
        # look for offsets
        srch_idx_start = rpeaks[i]
        srch_idx_end = tpeaks[i]
        if srch_idx_start is np.nan or srch_idx_end is np.nan:
            offsets.append(np.nan)
            continue
        dwt_local = dwtmatr[2 + degree, srch_idx_start:srch_idx_end]
        onset_slope_peaks, __ = scipy.signal.find_peaks(dwt_local)
        if len(onset_slope_peaks) == 0:
            offsets.append(np.nan)
            continue
        epsilon_offset = 0.5 * dwt_local[onset_slope_peaks[0]]
        if not (dwt_local[onset_slope_peaks[0] :] < epsilon_offset).any():
            offsets.append(np.nan)
            continue
        candidate_offsets = (
            np.where(dwt_local[onset_slope_peaks[0] :] < epsilon_offset)[0]
            + onset_slope_peaks[0]
        )
        offsets.append(candidate_offsets[0] + srch_idx_start)

        # # only for debugging
        # events_plot(candidate_offsets, dwt_local)
        # plt.plot(ecg[srch_idx_start: srch_idx_end], '--', label='ecg')
        # plt.legend()
        # plt.show()

    return onsets, offsets


def _dwt_compute_multiscales(ecg: np.ndarray, max_degree):
    """Return multiscales wavelet transforms."""

    def _apply_H_filter(signal_i, power=0):
        zeros = np.zeros(2**power - 1)
        timedelay = 2**power
        banks = np.r_[
            1.0 / 8,
            zeros,
            3.0 / 8,
            zeros,
            3.0 / 8,
            zeros,
            1.0 / 8,
        ]
        signal_f = scipy.signal.convolve(signal_i, banks, mode="full")
        signal_f[:-timedelay] = signal_f[timedelay:]  # timeshift: 2 steps
        return signal_f

    def _apply_G_filter(signal_i, power=0):
        zeros = np.zeros(2**power - 1)
        timedelay = 2**power
        banks = np.r_[2, zeros, -2]
        signal_f = scipy.signal.convolve(signal_i, banks, mode="full")
        signal_f[:-timedelay] = signal_f[timedelay:]  # timeshift: 1 step
        return signal_f

    dwtmatr = []
    intermediate_ret = np.array(ecg)
    for deg in range(max_degree):
        S_deg = _apply_G_filter(intermediate_ret, power=deg)
        T_deg = _apply_H_filter(intermediate_ret, power=deg)
        dwtmatr.append(S_deg)
        intermediate_ret = np.array(T_deg)
    dwtmatr = [
        arr[: len(ecg)] for arr in dwtmatr
    ]  # rescale transforms to the same length
    return np.array(dwtmatr)


# =============================================================================
# WAVELET METHOD (CWT)
# =============================================================================
def _ecg_delineator_cwt(ecg, rpeaks=None, sampling_rate=1000):
    # P-Peaks and T-Peaks
    tpeaks, ppeaks = _peaks_delineator(ecg, rpeaks, sampling_rate=sampling_rate)

    # qrs onsets and offsets
    qrs_onsets, qrs_offsets = _onset_offset_delineator(
        ecg, rpeaks, peak_type="rpeaks", sampling_rate=sampling_rate
    )

    # ppeaks onsets and offsets
    p_onsets, p_offsets = _onset_offset_delineator(
        ecg, ppeaks, peak_type="ppeaks", sampling_rate=sampling_rate
    )

    # tpeaks onsets and offsets
    t_onsets, t_offsets = _onset_offset_delineator(
        ecg, tpeaks, peak_type="tpeaks", sampling_rate=sampling_rate
    )

    # No dwt defined method for Q and S peak
    # Adopting manual method from "peak" method
    q_peaks = []
    s_peaks = []
    heartbeats = ecg_segment(ecg, rpeaks, sampling_rate=sampling_rate)
    for i, rpeak in enumerate(rpeaks):
        heartbeat = heartbeats[str(i + 1)]
        # Get index of R peaks
        R = heartbeat.index.get_loc(
            np.min(heartbeat.index.values[heartbeat.index.values > 0])
        )
        # Q wave
        Q_index, Q = _ecg_delineator_peak_Q(rpeak, heartbeat, R)
        q_peaks.append(Q_index)
        # S wave
        S_index, S = _ecg_delineator_peak_S(rpeak, heartbeat)
        s_peaks.append(S_index)

    # Return info dictionary
    return {
        "ECG_P_Onsets": p_onsets,
        "ECG_P_Peaks": ppeaks,
        "ECG_P_Offsets": p_offsets,
        "ECG_Q_Peaks": q_peaks,
        "ECG_R_Onsets": qrs_onsets,
        "ECG_R_Offsets": qrs_offsets,
        "ECG_S_Peaks": s_peaks,
        "ECG_T_Onsets": t_onsets,
        "ECG_T_Peaks": tpeaks,
        "ECG_T_Offsets": t_offsets,
    }


# =============================================================================
# PROMINENCE METHOD (Emrich et al., 2024)
# =============================================================================
def _prominence_ecg_delineator(ecg, rpeaks=None, sampling_rate=1000, **kwargs):
    # pysiology-informed boundaries in milliseconds, adapt if needed
    max_qrs_interval = int(kwargs.get("max_qrs_interval", 180) * sampling_rate / 1000)
    max_pr_interval = int(kwargs.get("max_pr_interval", 300) * sampling_rate / 1000)
    max_r_rise_time = int(kwargs.get("max_r_rise_time", 120) * sampling_rate / 1000)
    typical_st_segment = int(kwargs.get("typical_st_segment", 150) * sampling_rate / 1000)
    # max basepoint intervals
    max_p_basepoint_interval = int(kwargs.get("max_p_basepoint_interval", 100) * sampling_rate / 1000)
    max_r_basepoint_interval = int(kwargs.get("max_r_basepoint_interval", 100) * sampling_rate / 1000)
    max_t_basepoint_interval = int(kwargs.get("max_t_basepoint_interval", 200) * sampling_rate / 1000)

    waves = {
        "ECG_P_Onsets": [],
        "ECG_P_Peaks": [],
        "ECG_P_Offsets": [],
        "ECG_Q_Peaks": [],
        "ECG_R_Onsets": [],
        "ECG_R_Offsets": [],
        "ECG_S_Peaks": [],
        "ECG_T_Onsets": [],
        "ECG_T_Peaks": [],
        "ECG_T_Offsets": [],
    }

    # calculate RR intervals
    rr = np.diff(rpeaks)
    rr = np.insert(rr, 0, min(rr[0], 2 * rpeaks[0]))
    rr = np.insert(rr, -1, min(rr[-1], 2 * rpeaks[-1]))

    # iterate over all beats
    left = 0
    for i in range(len(rpeaks)):
        # 1. split signal into segments
        rpeak_pos = min(rpeaks[i], rr[i] // 2)
        left = rpeaks[i] - rpeak_pos
        right = rpeaks[i] + rr[i + 1] // 2
        ecg_seg = ecg[left:right]

        current_wave = {
            "ECG_R_Peaks": rpeak_pos,
        }

        # 2. find local extrema in signal
        local_maxima = scipy.signal.find_peaks(ecg_seg)[0]
        local_minima = scipy.signal.find_peaks(-ecg_seg)[0]
        local_extrema = np.concatenate((local_maxima, local_minima))

        # 3. compute prominence weight
        weight_maxima = _calc_prominence(local_maxima, ecg_seg, current_wave["ECG_R_Peaks"])
        weight_minima = _calc_prominence(local_minima, ecg_seg, current_wave["ECG_R_Peaks"], minima=True)

        if local_extrema.any():
            # find waves
            _prominence_find_q_wave(weight_minima, current_wave, max_r_rise_time)
            _prominence_find_s_wave(ecg_seg, weight_minima, current_wave, max_qrs_interval)
            _prominence_find_p_wave(local_maxima, weight_maxima, current_wave, max_pr_interval)
            _prominence_find_t_wave(local_extrema, (weight_minima + weight_maxima), current_wave, typical_st_segment)
            _prominence_find_on_offsets(
                ecg_seg,
                sampling_rate,
                local_minima,
                current_wave,
                max_p_basepoint_interval,
                max_r_basepoint_interval,
                max_t_basepoint_interval,
            )

        # append waves for current beat / complex
        for key in waves:
            if key == "ECG_R_Peaks":
                waves[key].append(int(rpeaks[i]))
            elif key in current_wave:
                waves[key].append(int(current_wave[key] + left))
            else:
                waves[key].append(np.nan)

    return waves


def _calc_prominence(peaks, sig, Rpeak=None, minima=False):
    """Returns an array of the same length as sig with prominences computed for the provided peaks.

    Prominence of the R-peak is excluded if the R-peak position is given.

    """
    w = np.zeros_like(sig)

    if len(peaks) < 1:
        return w
    # get prominence
    _sig = -sig if minima else sig
    w[peaks] = scipy.signal.peak_prominences(_sig, peaks)[0]
    # optional: set rpeak prominence to zero to emphasize other peaks
    if Rpeak is not None:
        w[Rpeak] = 0
    return w


def _prominence_find_q_wave(weight_minima, current_wave, max_r_rise_time):
    if "ECG_R_Peaks" not in current_wave:
        return
    q_bound = max(current_wave["ECG_R_Peaks"] - max_r_rise_time, 0)

    current_wave["ECG_Q_Peaks"] = np.argmax(weight_minima[q_bound : current_wave["ECG_R_Peaks"]]) + q_bound


def _prominence_find_s_wave(sig, weight_minima, current_wave, max_qrs_interval):
    if "ECG_Q_Peaks" not in current_wave:
        return
    s_bound = current_wave["ECG_Q_Peaks"] + max_qrs_interval
    s_wave = np.argmax(weight_minima[current_wave["ECG_R_Peaks"] : s_bound] > 0) + current_wave["ECG_R_Peaks"]
    current_wave["ECG_S_Peaks"] = (
        np.argmin(sig[current_wave["ECG_R_Peaks"] : s_bound]) + current_wave["ECG_R_Peaks"]
        if s_wave == current_wave["ECG_R_Peaks"]
        else s_wave
    )


def _prominence_find_p_wave(local_maxima, weight_maxima, current_wave, max_pr_interval):
    if "ECG_Q_Peaks" not in current_wave:
        return
    p_candidates = local_maxima[
        (current_wave["ECG_Q_Peaks"] - max_pr_interval <= local_maxima) & (local_maxima <= current_wave["ECG_Q_Peaks"])
    ]
    if p_candidates.any():
        current_wave["ECG_P_Peaks"] = p_candidates[np.argmax(weight_maxima[p_candidates])]


def _prominence_find_t_wave(local_extrema, weight_extrema, current_wave, typical_st_segment):
    if "ECG_S_Peaks" not in current_wave:
        return
    t_candidates = local_extrema[(current_wave["ECG_S_Peaks"] + typical_st_segment <= local_extrema)]
    if t_candidates.any():
        current_wave["ECG_T_Peaks"] = t_candidates[np.argmax(weight_extrema[t_candidates])]


def _prominence_find_on_offsets(
    sig,
    sampling_rate,
    local_minima,
    current_wave,
    max_p_basepoint_interval,
    max_r_basepoint_interval,
    max_t_basepoint_interval,
):
    if "ECG_P_Peaks" in current_wave:
        _, p_on, p_off = scipy.signal.peak_prominences(
            sig, [current_wave["ECG_P_Peaks"]], wlen=max_p_basepoint_interval
        )
        if not np.isnan(p_on):
            current_wave["ECG_P_Onsets"] = p_on[0]
        if not np.isnan(p_off):
            current_wave["ECG_P_Offsets"] = p_off[0]

    if "ECG_T_Peaks" in current_wave:
        p = -1 if np.isin(current_wave["ECG_T_Peaks"], local_minima) else 1

        _, t_on, t_off = scipy.signal.peak_prominences(
            p * sig, [current_wave["ECG_T_Peaks"]], wlen=max_t_basepoint_interval
        )
        if not np.isnan(t_on):
            current_wave["ECG_T_Onsets"] = t_on[0]
        if not np.isnan(t_off):
            current_wave["ECG_T_Offsets"] = t_off[0]

    # correct R-peak position towards local maxima (otherwise prominence will be falsely computed)
    r_pos = _correct_peak(sig, sampling_rate, current_wave["ECG_R_Peaks"])
    _, r_on, r_off = scipy.signal.peak_prominences(sig, [r_pos], wlen=max_r_basepoint_interval)
    if not np.isnan(r_on):
        current_wave["ECG_R_Onsets"] = r_on[0]

    if not np.isnan(r_off):
        current_wave["ECG_R_Offsets"] = r_off[0]


def _correct_peak(sig, fs, peak, window=0.02):
    """Correct peak towards local maxima within provided window."""

    left = peak - int(window * fs)
    right = peak + int(window * fs)
    if len(sig[left:right]) > 0:
        return np.argmax(sig[left:right]) + left
    else:
        return peak


# Internals
# ---------------------


def _onset_offset_delineator(ecg, peaks, peak_type="rpeaks", sampling_rate=1000):

    # first derivative of the Gaissian signal
    scales = np.array([1, 2, 4, 8, 16])
    cwtmatr, __ = pywt.cwt(ecg, scales, "gaus1", sampling_period=1.0 / sampling_rate)

    half_wave_width = int(0.1 * sampling_rate)  # NEED TO CHECK
    onsets = []
    offsets = []
    for index_peak in peaks:
        # find onset
        if np.isnan(index_peak):
            onsets.append(np.nan)
            offsets.append(np.nan)
            continue
        if peak_type == "rpeaks":
            search_window = cwtmatr[2, index_peak - half_wave_width : index_peak]
            prominence = 0.20 * max(search_window)
            height = 0.0
            wt_peaks, wt_peaks_data = scipy.signal.find_peaks(
                search_window, height=height, prominence=prominence
            )

        elif peak_type in ["tpeaks", "ppeaks"]:
            search_window = -cwtmatr[4, index_peak - half_wave_width : index_peak]

            prominence = 0.10 * max(search_window)
            height = 0.0
            wt_peaks, wt_peaks_data = scipy.signal.find_peaks(
                search_window, height=height, prominence=prominence
            )

        if len(wt_peaks) == 0:
            # print("Fail to find onset at index: %d", index_peak)
            onsets.append(np.nan)
        else:
            # The last peak is nfirst in (Martinez, 2004)
            nfirst = wt_peaks[-1] + index_peak - half_wave_width
            if peak_type == "rpeaks":
                if wt_peaks_data["peak_heights"][-1] > 0:
                    epsilon_onset = 0.05 * wt_peaks_data["peak_heights"][-1]
            elif peak_type == "ppeaks":
                epsilon_onset = 0.50 * wt_peaks_data["peak_heights"][-1]
            elif peak_type == "tpeaks":
                epsilon_onset = 0.25 * wt_peaks_data["peak_heights"][-1]
            leftbase = wt_peaks_data["left_bases"][-1] + index_peak - half_wave_width
            if peak_type == "rpeaks":
                candidate_onsets = np.where(cwtmatr[2, nfirst - 100 : nfirst] < epsilon_onset)[0] + nfirst - 100
            elif peak_type in ["tpeaks", "ppeaks"]:
                candidate_onsets = (
                    np.where(-cwtmatr[4, nfirst - 100 : nfirst] < epsilon_onset)[0]
                    + nfirst
                    - 100
                )

            candidate_onsets = candidate_onsets.tolist() + [leftbase]
            if len(candidate_onsets) == 0:
                onsets.append(np.nan)
            else:
                onsets.append(max(candidate_onsets))

        # find offset
        if peak_type == "rpeaks":
            search_window = -cwtmatr[2, index_peak : index_peak + half_wave_width]
            prominence = 0.50 * max(search_window)
            wt_peaks, wt_peaks_data = scipy.signal.find_peaks(
                search_window, height=height, prominence=prominence
            )

        elif peak_type in ["tpeaks", "ppeaks"]:
            search_window = cwtmatr[4, index_peak : index_peak + half_wave_width]
            prominence = 0.10 * max(search_window)
            wt_peaks, wt_peaks_data = scipy.signal.find_peaks(
                search_window, height=height, prominence=prominence
            )

        if len(wt_peaks) == 0:
            # print("Fail to find offsets at index: %d", index_peak)
            offsets.append(np.nan)
        else:
            nlast = wt_peaks[0] + index_peak
            epsilon_offset = 0  # Default value
            if peak_type == "rpeaks":
                if wt_peaks_data["peak_heights"][0] > 0:
                    epsilon_offset = 0.125 * wt_peaks_data["peak_heights"][0]
            elif peak_type == "ppeaks":
                epsilon_offset = 0.9 * wt_peaks_data["peak_heights"][0]
            elif peak_type == "tpeaks":
                epsilon_offset = 0.4 * wt_peaks_data["peak_heights"][0]
            rightbase = wt_peaks_data["right_bases"][0] + index_peak
            if peak_type == "rpeaks":
                candidate_offsets = (
                    np.where((-cwtmatr[2, nlast : nlast + 100]) < epsilon_offset)[0]
                    + nlast
                )
            elif peak_type in ["tpeaks", "ppeaks"]:
                candidate_offsets = (
                    np.where((cwtmatr[4, nlast : nlast + 100]) < epsilon_offset)[0]
                    + nlast
                )

            candidate_offsets = candidate_offsets.tolist() + [rightbase]
            if len(candidate_offsets) == 0:
                offsets.append(np.nan)
            else:
                offsets.append(min(candidate_offsets))

    onsets = np.array(onsets, dtype="object")
    offsets = np.array(offsets, dtype="object")
    return onsets, offsets


def _peaks_delineator(ecg, rpeaks, sampling_rate=1000):

    # first derivative of the Gaissian signal
    scales = np.array([1, 2, 4, 8, 16])
    cwtmatr, __ = pywt.cwt(ecg, scales, "gaus1", sampling_period=1.0 / sampling_rate)

    qrs_duration = 0.1

    search_boundary = int(0.9 * qrs_duration * sampling_rate / 2)
    significant_peaks_groups = []
    for i in range(len(rpeaks) - 1):
        # search for T peaks and P peaks from R peaks
        start = rpeaks[i] + search_boundary
        end = rpeaks[i + 1] - search_boundary
        search_window = cwtmatr[4, start:end]
        height = 0.25 * np.sqrt(np.mean(np.square(search_window)))
        peaks_tp, heights_tp = scipy.signal.find_peaks(
            np.abs(search_window), height=height
        )
        peaks_tp = peaks_tp + rpeaks[i] + search_boundary
        # set threshold for heights of peaks to find significant peaks in wavelet
        threshold = 0.125 * max(search_window)
        significant_peaks_tp = []
        significant_peaks_tp = [
            peaks_tp[j]
            for j in range(len(peaks_tp))
            if heights_tp["peak_heights"][j] > threshold
        ]

        significant_peaks_groups.append(
            _find_tppeaks(ecg, significant_peaks_tp, sampling_rate=sampling_rate)
        )

    tpeaks, ppeaks = zip(*[(g[0], g[-1]) for g in significant_peaks_groups])

    tpeaks = np.array(tpeaks, dtype="object")
    ppeaks = np.array(ppeaks, dtype="object")
    return tpeaks, ppeaks


def _find_tppeaks(ecg, keep_tp, sampling_rate=1000):

    # first derivative of the Gaissian signal
    scales = np.array([1, 2, 4, 8, 16])
    cwtmatr, __ = pywt.cwt(ecg, scales, "gaus1", sampling_period=1.0 / sampling_rate)
    max_search_duration = 0.05
    tppeaks = []
    for index_cur, index_next in zip(keep_tp[:-1], keep_tp[1:]):
        # limit 1
        correct_sign = (
            cwtmatr[4, :][index_cur] < 0 and cwtmatr[4, :][index_next] > 0
        )  # pylint: disable=R1716
        #    near = (index_next - index_cur) < max_wv_peak_dist #limit 2
        #    if near and correct_sign:
        if correct_sign:
            index_zero_cr = (
                signal_zerocrossings(cwtmatr[4, :][index_cur : index_next + 1])[0]
                + index_cur
            )
            nb_idx = int(max_search_duration * sampling_rate)
            index_max = np.argmax(
                ecg[index_zero_cr - nb_idx : index_zero_cr + nb_idx]
            ) + (index_zero_cr - nb_idx)
            tppeaks.append(index_max)
    if len(tppeaks) == 0:
        tppeaks = [np.nan]
    return tppeaks


# =============================================================================
#                              PEAK METHOD
# =============================================================================
def _ecg_delineator_peak(ecg, rpeaks=None, sampling_rate=1000):
    # Initialize
    heartbeats = ecg_segment(ecg, rpeaks, sampling_rate)

    Q_list = []
    P_list = []
    S_list = []
    T_list = []

    P_onsets = []
    T_offsets = []

    for i, rpeak in enumerate(rpeaks):
        heartbeat = heartbeats[str(i + 1)]

        # Get index of heartbeat
        R = heartbeat.index.get_loc(
            np.min(heartbeat.index.values[heartbeat.index.values > 0])
        )

        # Peaks ------
        # Q wave
        Q_index, Q = _ecg_delineator_peak_Q(rpeak, heartbeat, R)
        Q_list.append(Q_index)

        # P wave
        P_index, P = _ecg_delineator_peak_P(rpeak, heartbeat, R, Q)
        P_list.append(P_index)

        # S wave
        S_index, S = _ecg_delineator_peak_S(rpeak, heartbeat)
        S_list.append(S_index)

        # T wave
        T_index, T = _ecg_delineator_peak_T(rpeak, heartbeat, R, S)
        T_list.append(T_index)

        # Onsets/Offsets ------
        P_onsets.append(_ecg_delineator_peak_P_onset(rpeak, heartbeat, R, P))
        T_offsets.append(_ecg_delineator_peak_T_offset(rpeak, heartbeat, R, T))

    info = {
        "ECG_P_Peaks": P_list,
        "ECG_Q_Peaks": Q_list,
        "ECG_S_Peaks": S_list,
        "ECG_T_Peaks": T_list,
        "ECG_P_Onsets": P_onsets,
        "ECG_T_Offsets": T_offsets,
    }

    # Return info dictionary
    return info


# Internal
# --------------------------


def _ecg_delineator_peak_Q(rpeak, heartbeat, R):
    segment = heartbeat.loc[:0]  # Select left hand side

    Q = signal_findpeaks(
        -1 * segment["Signal"],
        height_min=0.05 * (segment["Signal"].max() - segment["Signal"].min()),
    )
    if len(Q["Peaks"]) == 0:
        return np.nan, None
    Q = Q["Peaks"][-1]  # Select most right-hand side
    from_R = R - Q  # Relative to R
    return rpeak - from_R, Q


def _ecg_delineator_peak_P(rpeak, heartbeat, R, Q):
    if Q is None:
        return np.nan, None

    segment = heartbeat.iloc[:Q]  # Select left of Q wave
    P = signal_findpeaks(
        segment["Signal"],
        height_min=0.05 * (segment["Signal"].max() - segment["Signal"].min()),
    )

    if len(P["Peaks"]) == 0:
        return np.nan, None
    P = P["Peaks"][np.argmax(P["Height"])]  # Select heighest
    from_R = R - P  # Relative to R
    return rpeak - from_R, P


def _ecg_delineator_peak_S(rpeak, heartbeat):
    segment = heartbeat.loc[0:]  # Select right hand side
    S = signal_findpeaks(
        -segment["Signal"],
        height_min=0.05 * (segment["Signal"].max() - segment["Signal"].min()),
    )

    if len(S["Peaks"]) == 0:
        return np.nan, None
    S = S["Peaks"][0]  # Select most left-hand side
    return rpeak + S, S


def _ecg_delineator_peak_T(rpeak, heartbeat, R, S):
    if S is None:
        return np.nan, None

    segment = heartbeat.iloc[R + S :]  # Select right of S wave
    T = signal_findpeaks(
        segment["Signal"],
        height_min=0.05 * (segment["Signal"].max() - segment["Signal"].min()),
    )

    if len(T["Peaks"]) == 0:
        return np.nan, None
    T = S + T["Peaks"][np.argmax(T["Height"])]  # Select heighest
    return rpeak + T, T


def _ecg_delineator_peak_P_onset(rpeak, heartbeat, R, P):
    if P is None:
        return np.nan

    segment = heartbeat.iloc[:P]  # Select left of P wave
    try:
        signal = signal_smooth(segment["Signal"].values, size=R / 10)
    except TypeError:
        signal = segment["Signal"]

    if len(signal) < 2:
        return np.nan

    signal = np.gradient(np.gradient(signal))
    P_onset = np.argmax(signal)

    from_R = R - P_onset  # Relative to R
    return rpeak - from_R


def _ecg_delineator_peak_T_offset(rpeak, heartbeat, R, T):
    if T is None:
        return np.nan

    segment = heartbeat.iloc[R + T :]  # Select right of T wave
    try:
        signal = signal_smooth(segment["Signal"].values, size=R / 10)
    except TypeError:
        signal = segment["Signal"]

    if len(signal) < 2:
        return np.nan

    signal = np.gradient(np.gradient(signal))
    T_offset = np.argmax(signal)

    return rpeak + T + T_offset


# =============================================================================
# Internals
# =============================================================================


def _ecg_delineate_plot(
    ecg_signal,
    rpeaks=None,
    signals=None,
    signal_features_type="all",
    sampling_rate=1000,
    window_start=-0.35,
    window_end=0.55,
    **kwargs
):
    """
    import neurokit2 as nk
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    ecg_signal = nk.data("ecg_100hz")

    # Extract R-peaks locations
     _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=1000)

    # Delineate the ECG signal with ecg_delineate()
    signals, waves = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=1000)

    # Plot the ECG signal with markings on ECG peaks
    _ecg_delineate_plot(ecg_signal, rpeaks=rpeaks, signals=signals,
                        signal_features_type='peaks', sampling_rate=1000)

    # Plot the ECG signal with markings on boundaries of R peaks
    _ecg_delineate_plot(ecg_signal, rpeaks=rpeaks, signals=signals,
                        signal_features_type='bound_R', sampling_rate=1000)

    # Plot the ECG signal with markings on boundaries of P peaks
    _ecg_delineate_plot(ecg_signal, rpeaks=rpeaks, signals=signals,
                        signal_features_type='bound_P', sampling_rate=1000)

    # Plot the ECG signal with markings on boundaries of T peaks
    _ecg_delineate_plot(ecg_signal, rpeaks=rpeaks, signals=signals,
                        signal_features_type='bound_T', sampling_rate=1000)

    # Plot the ECG signal with markings on all peaks and boundaries
    _ecg_delineate_plot(ecg_signal, rpeaks=rpeaks, signals=signals,
                        signal_features_type='all', sampling_rate=1000)

    """

    data = pd.DataFrame({"Signal": list(ecg_signal)})
    data = pd.concat([data, signals], axis=1)

    # Try retrieving right column
    if isinstance(rpeaks, dict):
        rpeaks = rpeaks["ECG_R_Peaks"]
    # Segment the signal around the R-peaks
    epochs = epochs_create(
        data,
        events=rpeaks,
        sampling_rate=sampling_rate,
        epochs_start=window_start,
        epochs_end=window_end,
    )
    data = epochs_to_df(epochs)
    data_cols = data.columns.values

    dfs = []
    for feature in data_cols:
        if signal_features_type == "peaks":
            if any(x in str(feature) for x in ["Peak"]):
                df = data[feature]
                dfs.append(df)
        elif signal_features_type == "bounds_R":
            if any(x in str(feature) for x in ["ECG_R_Onsets", "ECG_R_Offsets"]):
                df = data[feature]
                dfs.append(df)
        elif signal_features_type == "bounds_T":
            if any(x in str(feature) for x in ["ECG_T_Onsets", "ECG_T_Offsets"]):
                df = data[feature]
                dfs.append(df)
        elif signal_features_type == "bounds_P":
            if any(x in str(feature) for x in ["ECG_P_Onsets", "ECG_P_Offsets"]):
                df = data[feature]
                dfs.append(df)
        elif signal_features_type == "all":
            if any(x in str(feature) for x in ["Peak", "Onset", "Offset"]):
                df = data[feature]
                dfs.append(df)
    features = pd.concat(dfs, axis=1)

    fig, ax = plt.subplots()
    data.Label = data.Label.astype(int)
    for label in data.Label.unique():
        epoch_data = data[data.Label == label]
        ax.plot(epoch_data.Time, epoch_data.Signal, color="grey", alpha=0.2)
    for i, feature_type in enumerate(features.columns.values):  # pylint: disable=W0612
        event_data = data[data[feature_type] == 1.0]
        ax.scatter(
            event_data.Time, event_data.Signal, label=feature_type, alpha=0.5, s=200
        )
        ax.legend()
    return fig


def _ecg_delineate_check(waves, rpeaks):
    """This function replaces the delineated features with np.nan if its standardized distance from R-peaks is more than
    3."""
    df = pd.DataFrame.from_dict(waves)
    features_columns = df.columns

    df = pd.concat([df, pd.DataFrame({"ECG_R_Peaks": rpeaks})], axis=1)

    # loop through all columns to calculate the z distance
    for column in features_columns:  # pylint: disable=W0612
        df = _calculate_abs_z(df, features_columns)

    # Replace with nan if distance > 3
    for col in features_columns:
        for i in range(len(df)):
            if df["Dist_R_" + col][i] > 3:
                df[col][i] = np.nan

    # Return df without distance columns
    df = df[features_columns]
    waves = df.to_dict("list")
    return waves


def _calculate_abs_z(df, columns):
    """This function helps to calculate the absolute standardized distance between R-peaks and other delineated waves
    features by `ecg_delineate()`"""
    for column in columns:
        df["Dist_R_" + column] = np.abs(
            standardize(df[column].sub(df["ECG_R_Peaks"], axis=0))
        )
    return df

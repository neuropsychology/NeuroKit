# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..events import events_find
from ..misc import as_vector
from ..signal import (
    signal_binarize,
    signal_changepoints,
    signal_formatpeaks,
    signal_smooth,
)


def emg_activation(
    emg_amplitude=None,
    emg_cleaned=None,
    sampling_rate=1000,
    method="threshold",
    threshold="default",
    duration_min="default",
    size=None,
    threshold_size=None,
    **kwargs,
):
    """**Locate EMG Activity**

    Detects onset in EMG signal based on the amplitude threshold.

    Parameters
    ----------
    emg_amplitude : array
        At least one EMG-related signal. Either the amplitude of the EMG signal, obtained from
        ``emg_amplitude()`` for methods like ``"threshold"`` or ``"mixture"``), and / or the
        cleaned EMG signal (for methods like ``"pelt"``, ``"biosppy"`` or ``"silva"``).
    emg_cleaned : array
        At least one EMG-related signal. Either the amplitude of the EMG signal, obtained from
        ``emg_amplitude()`` for methods like ``"threshold"`` or ``"mixture"``), and / or the
        cleaned EMG signal (for methods like ``"pelt"``, ``"biosppy"`` or ``"silva"``).
    sampling_rate : int
        The sampling frequency of ``emg_signal`` (in Hz, i.e., samples/second).
    method : str
        The algorithm used to discriminate between activity and baseline. Can be one of
        ``"mixture"`` (default) or ``"threshold"``. If ``"mixture"``, will use a Gaussian Mixture
        Model to categorize between the two states. If ``"threshold"``, will consider as activated
        all points which amplitude is superior to the threshold. Can also be ``"pelt"`` or
        ``"biosppy"`` or ``"silva"``.
    threshold : str
        If ``method`` is ``"mixture"``, then it corresponds to the minimum probability required to
        be considered as activated (default to 0.33). If ``method`` is ``"threshold"``, then it
        corresponds to the minimum amplitude to detect as onset i.e., defaults to one tenth of the
        standard deviation of ``emg_amplitude``. If ``method`` is ``"silva"``, defaults to 0.05. If
        ``method`` is ``"biosppy"``, defaults to 1.2 times of the mean of the absolute of the
        smoothed, full-wave-rectified signal. If ``method`` is ``"pelt"``, threshold defaults to
        ``None`` as changepoints are used as a basis for detection.
    duration_min : float
        The minimum duration of a period of activity or non-activity in seconds.
        If ``default``, will be set to 0.05 (50 ms).
    size: float or int
        Detection window size (seconds). Applicable only if ``method`` is ``"biosppy"`` or
        ``"silva"``. If ``None``, defaults to 0.05 for ``"biosppy"`` and 20 for ``"silva"``.
    threshold_size : int
        Window size for calculation of the adaptive threshold. Must be bigger than the detection
        window size. Applicable only if ``method`` is ``"silva``". If ``None``, defaults to 22.
    kwargs : optional
        Other arguments.


    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the samples at which the
        onsets, offsets, and periods of activations of the EMG signal occur, accessible with the
        key ``"EMG_Onsets"``, ``"EMG_Offsets"``, and ``"EMG_Activity"`` respectively.
    activity_signal : DataFrame
        A DataFrame of same length as the input signal in which occurences of onsets, offsets, and
        activity (above the threshold) of the EMG signal are marked as "1" in lists of zeros with
        the same length as ``emg_amplitude``. Accessible with the keys ``"EMG_Onsets"``,
        ``"EMG_Offsets"``, and ``"EMG_Activity"`` respectively.

    See Also
    --------
    emg_simulate, emg_clean, emg_amplitude, emg_process, emg_plot

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate signal and obtain amplitude
      emg = nk.emg_simulate(duration=10, burst_number=3)
      emg_cleaned = nk.emg_clean(emg)
      emg_amplitude = nk.emg_amplitude(emg_cleaned)

    * **Example 1:** Threshold method

    .. ipython:: python

      activity, info = nk.emg_activation(emg_amplitude=emg_amplitude, method="threshold")

      @savefig p_emg_activation1.png scale=100%
      nk.events_plot([info["EMG_Offsets"], info["EMG_Onsets"]], emg_cleaned)
      @suppress
      plt.close()

    * **Example 2:** Pelt method

    .. ipython:: python

      activity, info = nk.emg_activation(emg_cleaned=emg_cleaned, method="pelt")
      @savefig p_emg_activation2.png scale=100%
      nk.events_plot([info["EMG_Offsets"], info["EMG_Onsets"]], emg_cleaned)
      @suppress
      plt.close()

    * **Example 3:** Biosppy method

    .. ipython:: python

      activity, info = nk.emg_activation(emg_cleaned=emg_cleaned, method="biosppy")
      @savefig p_emg_activation3.png scale=100%
      nk.events_plot([info["EMG_Offsets"], info["EMG_Onsets"]], emg_cleaned)
      @suppress
      plt.close()

    * **Example 4:** Silva method

    .. ipython:: python

      activity, info = nk.emg_activation(emg_cleaned=emg_cleaned, method="silva")
      @savefig p_emg_activation4.png scale=100%
      nk.events_plot([info["EMG_Offsets"], info["EMG_Onsets"]], emg_cleaned)
      @suppress
      plt.close()


    References
    ----------
    * Silva H, Scherer R, Sousa J, Londral A , "Towards improving the ssability of
      electromyographic interfacess", Journal of Oral Rehabilitation, pp. 1-2, 2012.

    """
    # Sanity checks.
    if emg_amplitude is not None:
        emg_amplitude = as_vector(emg_amplitude)
    if emg_cleaned is not None:
        emg_cleaned = as_vector(emg_cleaned)
        if emg_amplitude is None:
            emg_amplitude = as_vector(emg_cleaned)

    if duration_min == "default":
        duration_min = int(0.05 * sampling_rate)

    # Find offsets and onsets.
    method = method.lower()  # remove capitalised letters
    if method == "threshold":
        if emg_amplitude is None:
            raise ValueError(
                "NeuroKit error: emg_activation(): 'threshold' method needs 'emg_amplitude' signal to be passed."
            )
        activity = _emg_activation_threshold(emg_amplitude, threshold=threshold)

    elif method == "mixture":
        if emg_amplitude is None:
            raise ValueError(
                "NeuroKit error: emg_activation(): 'mixture' method needs 'emg_amplitude' signal to be passed."
            )
        activity = _emg_activation_mixture(emg_amplitude, threshold=threshold)

    elif method == "pelt":
        if emg_cleaned is None:
            raise ValueError(
                "NeuroKit error: emg_activation(): 'pelt' method needs 'emg_cleaned' (cleaned or raw EMG) signal to "
                "be passed."
            )
        activity = _emg_activation_pelt(
            emg_cleaned, duration_min=duration_min, **kwargs
        )

    elif method == "biosppy":
        if emg_cleaned is None:
            raise ValueError(
                "NeuroKit error: emg_activation(): 'biosppy' method needs 'emg_cleaned' (cleaned EMG) "
                "signal to be passed."
            )
        if size is None:
            size = 0.05
        activity = _emg_activation_biosppy(
            emg_cleaned, sampling_rate=sampling_rate, size=size, threshold=threshold
        )

    elif method == "silva":
        if emg_cleaned is None:
            raise ValueError(
                "NeuroKit error: emg_activation(): 'silva' method needs 'emg_cleaned' (cleaned EMG) "
                "signal to be passed."
            )
        if size is None:
            size = 20
        if threshold_size is None:
            threshold_size = 22
        activity = _emg_activation_silva(
            emg_cleaned, size=size, threshold=threshold, threshold_size=threshold_size
        )

    else:
        raise ValueError(
            "NeuroKit error: emg_activation(): 'method' should be one of 'mixture', 'threshold', 'pelt' or 'biosppy'."
        )

    # Sanitize activity.
    info = _emg_activation_activations(activity, duration_min=duration_min)

    # Prepare Output.
    df_activity = signal_formatpeaks(
        {"EMG_Activity": info["EMG_Activity"]},
        desired_length=len(emg_amplitude),
        peak_indices=info["EMG_Activity"],
    )
    df_onsets = signal_formatpeaks(
        {"EMG_Onsets": info["EMG_Onsets"]},
        desired_length=len(emg_amplitude),
        peak_indices=info["EMG_Onsets"],
    )
    df_offsets = signal_formatpeaks(
        {"EMG_Offsets": info["EMG_Offsets"]},
        desired_length=len(emg_amplitude),
        peak_indices=info["EMG_Offsets"],
    )

    # Modify output produced by signal_formatpeaks.
    for x in range(len(emg_amplitude)):
        if df_activity.loc[x, "EMG_Activity"] != 0:
            if df_activity.index[x] == df_activity.index.get_loc(x):
                df_activity.loc[x, "EMG_Activity"] = 1
            else:
                df_activity.loc[x, "EMG_Activity"] = 0
        if df_offsets.loc[x, "EMG_Offsets"] != 0:
            if df_offsets.index[x] == df_offsets.index.get_loc(x):
                df_offsets.loc[x, "EMG_Offsets"] = 1
            else:
                df_offsets.loc[x, "EMG_Offsets"] = 0

    activity_signal = pd.concat([df_activity, df_onsets, df_offsets], axis=1)

    return activity_signal, info


# =============================================================================
# Methods
# =============================================================================


def _emg_activation_threshold(emg_amplitude, threshold="default"):
    if threshold == "default":
        threshold = (1 / 10) * np.std(emg_amplitude)

    if threshold > np.max(emg_amplitude):
        raise ValueError(
            "NeuroKit error: emg_activation(): the threshold specified exceeds the maximum of the signal"
            "amplitude."
        )

    activity = signal_binarize(emg_amplitude, method="threshold", threshold=threshold)
    return activity


def _emg_activation_mixture(emg_amplitude, threshold="default"):
    if threshold == "default":
        threshold = 0.33

    activity = signal_binarize(emg_amplitude, method="mixture", threshold=threshold)
    return activity


def _emg_activation_pelt(emg_cleaned, threshold="default", duration_min=0.05, **kwargs):
    if threshold == "default":
        threshold = None

    # Get changepoints
    changepoints = signal_changepoints(emg_cleaned, change="var", show=False, **kwargs)

    # Add first point
    if changepoints[0] != 0:
        changepoints = np.append(0, changepoints)

    # Sanitize
    lengths = np.append(0, np.diff(changepoints))
    changepoints = changepoints[1:][lengths[1:] > duration_min]

    # reèAdd first point
    if changepoints[0] != 0:
        changepoints = np.append(0, changepoints)

    binary = np.full(len(emg_cleaned), np.nan)
    binary[changepoints[0::2]] = 0
    binary[changepoints[1::2]] = 1

    activity = pd.Series(binary).ffill().values

    # Label as 1 to parts that have the larger SD (likely to be activations)
    if emg_cleaned[activity == 1].std() > emg_cleaned[activity == 0].std():
        activity = np.abs(activity - 1)

    activity[0] = 0
    activity[-1] = 0

    return activity


def _emg_activation_biosppy(
    emg_cleaned, sampling_rate=1000, size=0.05, threshold="default"
):
    """Adapted from `find_onsets` in Biosppy."""

    # check inputs
    if emg_cleaned is None:
        raise TypeError("Please specify an input signal.")

    # full-wave rectification
    fwlo = np.abs(emg_cleaned)

    # smooth
    size = int(sampling_rate * size)
    mvgav = signal_smooth(fwlo, method="convolution", kernel="boxzen", size=size)

    # threshold
    if threshold == "default":
        aux = np.abs(mvgav)
        threshold = 1.2 * np.mean(aux)

    # find onsets
    # length = len(signal)
    # start = np.nonzero(mvgav > threshold)[0]
    # stop = np.nonzero(mvgav <= threshold)[0]

    # onsets = np.union1d(np.intersect1d(start - 1, stop),
    #                     np.intersect1d(start + 1, stop))

    # if np.any(onsets):
    #     if onsets[-1] >= length:
    #         onsets[-1] = length - 1

    activity = signal_binarize(mvgav, method="threshold", threshold=threshold)

    return activity


def _emg_activation_silva(emg_cleaned, size=20, threshold_size=22, threshold="default"):
    """Follows the approach by Silva et al. 2012, adapted from `Biosppy`."""

    if threshold_size <= size:
        raise ValueError(
            "NeuroKit error: emg_activation(): The window size for calculation of the "
            "adaptive threshold must be bigger than the detection window size."
        )

    if threshold == "default":
        threshold = 0.05

    # subtract baseline offset
    signal_zero_mean = emg_cleaned - np.mean(emg_cleaned)

    # full-wave rectification
    fwlo = np.abs(signal_zero_mean)

    # moving average for calculating the test function
    tf_mvgav = np.convolve(fwlo, np.ones((size,)) / size, mode="valid")

    # moving average for calculating the adaptive threshold
    threshold_mvgav = np.convolve(
        fwlo, np.ones((threshold_size,)) / threshold_size, mode="valid"
    )

    onset_time_list = []
    offset_time_list = []
    onset = False
    for k in range(0, len(threshold_mvgav)):
        if onset is True:
            # an onset was previously detected, look for offset time
            if tf_mvgav[k] < threshold_mvgav[k] and tf_mvgav[k] < threshold:
                offset_time_list.append(k)
                onset = False
                # the offset has been detected, and we can look for another activation
        else:
            # we only look for another onset if a previous offset was detected
            if tf_mvgav[k] >= threshold_mvgav[k] and tf_mvgav[k] >= threshold:
                onset_time_list.append(k)
                onset = True

    onsets = np.union1d(onset_time_list, offset_time_list)

    # adjust indices because of moving average
    onsets += int(size / 2)

    binary = np.full(len(emg_cleaned), np.nan)
    binary[onsets[0::2]] = 0
    binary[onsets[1::2]] = 1

    activity = pd.Series(binary).bfill().values
    activity = pd.Series(activity).fillna(0)

    return activity


# =============================================================================
# Internals
# =============================================================================
def _emg_activation_activations(activity, duration_min=0.05):
    activations = events_find(
        activity, threshold=0.5, threshold_keep="above", duration_min=duration_min
    )
    activations["offset"] = activations["onset"] + activations["duration"]

    baseline = events_find(
        activity == 0, threshold=0.5, threshold_keep="above", duration_min=duration_min
    )
    baseline["offset"] = baseline["onset"] + baseline["duration"]

    # Cross-comparison
    valid = np.isin(activations["onset"], baseline["offset"])
    onsets = activations["onset"][valid]
    offsets = activations["offset"][valid]

    # make sure offset indices are within length of signal
    offsets = offsets[offsets < len(activity)]

    new_activity = np.array([])
    for x, y in zip(onsets, offsets):
        activated = np.arange(x, y)
        new_activity = np.append(new_activity, activated)

    # Prepare Output.
    info = {"EMG_Onsets": onsets, "EMG_Offsets": offsets, "EMG_Activity": new_activity}

    return info

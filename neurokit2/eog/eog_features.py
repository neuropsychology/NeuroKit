import numpy as np
import pandas as pd

from ..epochs import epochs_create
from ..signal import signal_zerocrossings


def eog_features(eog_cleaned, peaks, sampling_rate=1000):
    """**Extracts Features of EOG Eye Blinks**

    Extracts features of EOG eye blinks e.g., velocity measures, blink-amplitude-ratio (BAR),
    duration, and markers of onset and offset of each blink.

    The positive amplitude velocity ratio (pAVR) and the negative amplitude velocity ratio (nAVR).
    The positive amplitude velocity ratio is the ratio of the maximum amplitude of the blink over
    the maximum velocity (rate of change) during the blink upStroke. Similarly, the negative
    amplitude velocity ratio is the ratio of the maximum amplitude of the blink over the maximum
    velocity found in the blink downStroke. These measures have units of centiseconds and are
    indicators of fatigue.

    The blink-amplitude ratio (BAR) is the average amplitude of the signal between the blink
    leftZero and rightZero zero crossings divided by the average amplitude of the positive fraction
    of the signal “outside” the blink. BAR values in the range [5, 20]. BAR is a measure of the
    signal-to-noise ratio (SNR) of the blink to the background in a candidate signal.

    Parameters
    ----------
    eog_cleaned : Union[list, np.array, pd.Series]
        The cleaned EOG channel, extracted from :func:`.eog_clean`.
    peaks : np.array
        Vector containing the samples at which EOG-peaks occur.
    sampling_rate : int
        The sampling frequency of :func:`.eog_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.

    Returns
    -------
    info : dict
        A dictionary containing information of the features of the EOG blinks, accessible with keys
        ``"Blink_LeftZeros"`` (point when eye closes), ``"Blink_RightZeros"`` (point when eye opens)
        , ``"Blink_pAVR"``, ``"Blink_nAVR"``, ``"Blink_BAR"``, and ``"Blink_Duration"`` (duration
        of each blink in seconds).

    See Also
    --------
    eog_clean, eog_findpeaks

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Get data
      eog_signal = nk.data('eog_100hz')
      eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=100)
      peaks = nk.eog_findpeaks(eog_cleaned, sampling_rate=100)
      info = nk.eog_features(eog_cleaned, peaks, sampling_rate=100)

    References
    ----------
    * Kleifges, K., Bigdely-Shamlo, N., Kerick, S. E., & Robbins, K. A. (2017). BLINKER: automated
      extraction of ocular indices from EEG enabling large-scale analysis. Frontiers in
      neuroscience, 11, 12.

    """

    if len(peaks) == 0:
        return {
            "Blink_LeftZeros": [],
            "Blink_RightZeros": [],
            "Blink_pAVR": [],
            "Blink_nAVR": [],
            "Blink_BAR": [],
            "Blink_Duration": [],
        }

    BARs, _, leftzeros, rightzeros, downstrokes, upstrokes = _eog_features_delineate(
        eog_cleaned, peaks, sampling_rate=sampling_rate
    )

    pAVR_list = []
    nAVR_list = []
    duration_list = []

    for i in range(len(peaks)):
        # Closing blink (pAVR): ratio of max blink amplitude over max upstroke velocity (in cs)
        blink_close = upstrokes[i].Signal.to_numpy()
        max_amp = blink_close[-1] if len(blink_close) > 0 else np.nan
        vel_close = np.diff(blink_close) * sampling_rate
        pAVR = (max_amp / np.max(vel_close)) * 100 if len(vel_close) > 0 and np.max(vel_close) > 0 else np.nan
        pAVR_list.append(pAVR)

        # Opening blink (nAVR): ratio of max blink amplitude over max downstroke velocity (in cs)
        blink_open = downstrokes[i].Signal.to_numpy()
        vel_open = -np.diff(blink_open) * sampling_rate
        nAVR = (max_amp / np.max(vel_open)) * 100 if len(vel_open) > 0 and np.max(vel_open) > 0 else np.nan
        nAVR_list.append(nAVR)

        # Duration (in seconds)
        duration_full = (rightzeros[i] - leftzeros[i]) / sampling_rate
        duration_list.append(duration_full)

    # Return info dictionary
    info = {
        "Blink_LeftZeros": leftzeros,
        "Blink_RightZeros": rightzeros,
        "Blink_pAVR": pAVR_list,
        "Blink_nAVR": nAVR_list,
        "Blink_BAR": BARs,
        "Blink_Duration": duration_list,
    }

    return info


# =============================================================================
# Internals
# =============================================================================


def _eog_features_delineate(eog_cleaned, candidates, sampling_rate=1000):
    # Calculate blink landmarks
    epochs = epochs_create(
        eog_cleaned,
        events=candidates,
        sampling_rate=sampling_rate,
        epochs_start=-0.5,
        epochs_end=0.5,
    )

    # max value marker
    peaks = []
    leftzeros = []
    rightzeros = []
    downstrokes = []
    upstrokes = []
    BARs = []

    for i in epochs:
        max_value = epochs[i].Signal.max()

        # Check if peak is at the end or start of epoch
        t = epochs[i].loc[epochs[i]["Signal"] == max_value].index
        if np.all((t > 0.3) & (t < 0.51)):
            # Trim end of epoch
            epochs[i] = epochs[i][-0.5:0.3]
            max_value = epochs[i].Signal.max()
        if np.all((t > -0.51) & (t < -0.3)):
            # Trim start of epoch
            epochs[i] = epochs[i][-0.3:0.5]
            max_value = epochs[i].Signal.max()

        # Find position of peak
        max_frame = epochs[i]["Index"].loc[epochs[i]["Signal"] == max_value]
        max_frame = np.asarray(max_frame)[0].item()  # If two points achieve max value, first one is blink

        # left and right zero markers
        crossings = signal_zerocrossings(epochs[i].Signal)
        crossings_idx = np.asarray(epochs[i]["Index"].iloc[crossings])
        crossings_idx = np.unique(np.append(crossings_idx, max_frame))
        max_position = np.where(crossings_idx == max_frame)[0][0].item()

        if (max_position - 1) >= 0:  # crosses zero point
            leftzero = crossings_idx[max_position - 1]
        else:
            max_value_t = epochs[i].Signal.idxmax()
            sliced_before = epochs[i].loc[slice(max_value_t), :]
            leftzero = sliced_before["Index"].loc[sliced_before["Signal"] == sliced_before["Signal"].min()]
            leftzero = np.asarray(leftzero)[0].item()

        if (max_position + 1) < len(crossings_idx):  # crosses zero point
            rightzero = crossings_idx[max_position + 1]
        else:
            max_value_t = epochs[i].Signal.idxmax()
            sliced_after = epochs[i].loc[slice(max_value_t, None), :]
            rightzero = sliced_after["Index"].loc[sliced_after["Signal"] == sliced_after["Signal"].min()]
            rightzero = np.asarray(rightzero)[0].item()

        # upstroke and downstroke markers (inclusive of peak and zero crossings)
        upstroke_idx = list(np.arange(leftzero, max_frame + 1))
        upstroke = epochs[i].loc[epochs[i]["Index"].isin(upstroke_idx)]
        downstroke_idx = list(np.arange(max_frame, rightzero + 1))
        downstroke = epochs[i].loc[epochs[i]["Index"].isin(downstroke_idx)]

        # left base and right base markers
        leftbase_idx = list(np.arange(epochs[i]["Index"].iloc[0], leftzero))
        leftbase_signal = epochs[i].loc[epochs[i]["Index"].isin(leftbase_idx)]
        #        leftbase_min = leftbase_signal['Signal'].min()
        #        leftbase = np.array(leftbase_signal['Index'].loc[leftbase_signal['Signal'] == leftbase_min])[0]

        rightbase_idx = list(np.arange(rightzero, epochs[i]["Index"].iloc[epochs[i].shape[0] - 1]))
        rightbase_signal = epochs[i].loc[epochs[i]["Index"].isin(rightbase_idx)]
        #        rightbase_min = rightbase_signal['Signal'].min()
        #        rightbase = np.array(rightbase_signal['Index'].loc[rightbase_signal['Signal'] == rightbase_min])[0]

        # Rejecting candidate signals with low SNR (BAR = blink-amplitude-ratio)
        inside_blink_idx = list(np.arange(leftzero, rightzero))
        inside_blink = epochs[i].loc[epochs[i]["Index"].isin(inside_blink_idx)]
        outside_blink = pd.concat([leftbase_signal, rightbase_signal], axis=0)

        BAR = inside_blink.Signal.mean() / outside_blink.Signal[outside_blink["Signal"] > 0].mean()

        # Features of all candidates
        BARs.append(BAR)
        leftzeros.append(leftzero)
        rightzeros.append(rightzero)
        downstrokes.append(downstroke)
        upstrokes.append(upstroke)

        # BAR values in the range [5, 20] usually capture blinks reasonably well
        if 3 < BAR < 50:
            peaks.append(max_frame)

    return BARs, peaks, leftzeros, rightzeros, downstrokes, upstrokes

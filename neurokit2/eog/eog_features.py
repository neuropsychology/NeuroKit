# -*- coding: utf-8 -*-
import numpy as np

from ..epochs import epochs_create
from ..signal import signal_zerocrossings


def eog_features(eog_cleaned, peaks, sampling_rate):
    """nAVRZ: Statistics for the negative amplitude velocity ratio calculated from the maxFrame to the rightZero of each blink.
    pAVRZ: Statistics for the positive amplitude velocity ratio calculated from the leftZero to maxFrame of each blink.

    """
    _, _, downstrokes, upstrokes = _eog_features_delineate(eog_cleaned, peaks, sampling_rate=sampling_rate)

    # Closing blink (pAVR)
    blink_close = eog_cleaned[upstrokes[0]]
    change_close = np.diff(blink_close)
    duration_close = len(change_close)/sampling_rate
    pAVR = abs(change_close.max()/duration_close)*100

    # Opening blink (nAVR)
    blink_open = eog_cleaned[downstrokes[0]]
    change_open = np.diff(blink_open)
    duration_open = len(change_open)/sampling_rate
    nAVR = abs(change_open.max()/duration_open)*100

    return pAVR, nAVR


def _eog_features_delineate(eog_cleaned, peaks, sampling_rate=1000):
    """Computes EOG Rate (Number of blinks per minute).

    The amplitude-velocity ratio introduced by Johns (2003) relates directly to drowsiness. The pAVR is
    the ratio of the maximum signal amplitude to the maximum eye-closing signal velocity for the blink,
    while the nAVR is the ratio of the maximum signal amplitude to the maximum eye-opening signal velocity
    for the blink. Both ratios are in units of time, independent of the units of the amplitude.

    """

    # Calculate blink landmarks
    epochs = epochs_create(
        eog_cleaned, events=peaks, sampling_rate=sampling_rate, epochs_start=-0.5, epochs_end=0.5
    )

    leftzeros = []
    rightzeros = []
    downstrokes = []
    upstrokes = []
    for i in epochs:
        max_value = epochs[i].Signal.max()

        # Check if peak is at the end or start of epoch
        t = epochs[i].loc[epochs[i]["Signal"] == max_value].index
        if np.all(0.35 < t < 0.51):
            # Trim end of epoch
            epochs[i] = epochs[i][-0.5:0.35]
            max_value = epochs[i].Signal.max()
        if np.all(-0.51 < t < -0.35):
            # Trim start of epoch
            epochs[i] = epochs[i][-0.35:0.5]
            max_value = epochs[i].Signal.max()

        # Find position of peak
        max_frame = epochs[i]["Index"].loc[epochs[i]["Signal"] == max_value]
        max_frame = np.array(max_frame)
        if len(max_frame) > 1:
            max_frame = max_frame[0]  # If two points achieve max value, first one is blink
        else:
            max_frame = int(max_frame)

        # left and right zero markers
        crossings = signal_zerocrossings(epochs[i].Signal)
        crossings_idx = epochs[i]["Index"].iloc[crossings]
        crossings_idx = np.sort(np.append([np.array(crossings_idx)], [max_frame]))
        max_position = int(np.where(crossings_idx == max_frame)[0])

        if (max_position - 1) >= 0:  # crosses zero point
            leftzero = crossings_idx[max_position - 1]
        else:
            max_value_t = epochs[i].Signal.idxmax()
            sliced_before = epochs[i].loc[slice(max_value_t), :]
            leftzero = sliced_before["Index"].loc[sliced_before["Signal"] == sliced_before["Signal"].min()]
            leftzero = np.array(leftzero)[0]

        if (max_position + 1) < len(crossings_idx):  # crosses zero point
            rightzero = crossings_idx[max_position + 1]
        else:
            max_value_t = epochs[i].Signal.idxmax()
            sliced_before = epochs[i].loc[slice(max_value_t), :]
            sliced_after = epochs[i].tail(epochs[i].shape[0] - sliced_before.shape[0])
            rightzero = sliced_after["Index"].loc[sliced_after["Signal"] == sliced_after["Signal"].min()]
            rightzero = np.array(rightzero)[0]

        # upstroke and downstroke markers
        upstroke_idx = list(np.arange(leftzero, max_frame))
        upstroke = epochs[i].loc[epochs[i]['Index'].isin(upstroke_idx)]
        downstroke_idx = list(np.arange(max_frame, rightzero))
        downstroke = epochs[i].loc[epochs[i]['Index'].isin(downstroke_idx)]

        # left base and right base markers
        leftbase_idx = list(np.arange(epochs[i]["Index"].iloc[0], leftzero))
        leftbase_signal = epochs[i].loc[epochs[i]["Index"].isin(leftbase_idx)]
        #        leftbase_min = leftbase_signal['Signal'].min()
        #        leftbase = np.array(leftbase_signal['Index'].loc[leftbase_signal['Signal'] == leftbase_min])[0]

        rightbase_idx = list(np.arange(rightzero, epochs[i]["Index"].iloc[epochs[i].shape[0] - 1]))
        rightbase_signal = epochs[i].loc[epochs[i]["Index"].isin(rightbase_idx)]
        #        rightbase_min = rightbase_signal['Signal'].min()
        #        rightbase = np.array(rightbase_signal['Index'].loc[rightbase_signal['Signal'] == rightbase_min])[0]

        # Get features
        downstrokes.append(downstroke_idx)
        upstrokes.append(upstroke_idx)
        leftzeros.append(leftzero)
        rightzeros.append(rightzero)

    return leftzeros, rightzeros, downstrokes, upstrokes


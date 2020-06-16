# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..epochs import epochs_create
from ..signal import signal_filter, signal_zerocrossings


def _eog_blinks_landmarks(eog_signal, sampling_rate):
    """
    Example
    -------
    >>> # Load signal
    >>> raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() +
    ...                           '/MEG/sample/sample_audvis_raw.fif', preload=True) #doctest: +SKIP
    >>> eog_signal = nk.mne_channel_extract(raw, what='EOG', name='EOG')*-1
    >>> sampling_rate = raw.info['sfreq']

    >>> # Extract blink landmarks
    >>> peaks = _eog_blinks_landmarks(eog_signal, sampling_rate)
    >>> events = [[i] for i in peaks]
    >>> fig = nk.events_plot(events, eog_filtered)  # doctest: +ELLIPSIS
    >>> fig
    """
    # bandpass filter prior to blink detection
    eog_filtered = signal_filter(eog_signal, sampling_rate, lowcut=1, highcut=20)

    # Establish criterion
    threshold = 1.5 * np.std(eog_filtered) + eog_filtered.mean()
    min_blink = 0.05 * sampling_rate  # min blink frames

    blink = np.full(len(eog_filtered), False, dtype=bool)
    index = []
    for i in range(len(eog_filtered)):
        if eog_filtered[i] > threshold:
            index.append(i)
            blink[i] = True

    candidates = np.array(index)[np.where(np.diff(index) > min_blink)[0]]

    # Calculate blink landmarks
    epochs = epochs_create(eog_filtered, events=candidates, sampling_rate=sampling_rate,
                           epochs_start=-0.5, epochs_end=0.5)

    # max value marker
    candidate_blinks = []
    peaks = []
    for i in epochs:
        max_value = epochs[i].Signal.max()

        # Check if peak is at the end or start of epoch
        t = epochs[i].loc[epochs[i]['Signal'] == max_value].index
        if 0.3 < t < 0.51:
            # Trim end of epoch
            epochs[i] = epochs[i][-0.5:0.3]
            max_value = epochs[i].Signal.max()
        if -0.51 < t < -0.3:
            # Trim start of epoch
            epochs[i] = epochs[i][-0.3:0.5]
            max_value = epochs[i].Signal.max()

        # Find position of peak
        max_frame = epochs[i]['Index'].loc[epochs[i]['Signal'] == max_value]
        max_frame = np.array(max_frame)
        if len(max_frame) > 1:
            max_frame = max_frame[0]  # If two points achieve max value, first one is blink
        else:
            max_frame = int(max_frame)

        # left and right zero markers
        crossings = signal_zerocrossings(epochs[i].Signal)
        crossings_idx = epochs[i]['Index'].iloc[crossings]
        crossings_idx = np.sort(np.append([np.array(crossings_idx)], [max_frame]))
        max_position = int(np.where(crossings_idx == max_frame)[0])

        leftzero = crossings_idx[max_position-1]
        rightzero = crossings_idx[max_position+1]

        max_value_t = epochs[i].Signal.idxmax()
        sliced_before = epochs[i].loc[slice(max_value_t), :]
        sliced_after = epochs[i].tail(epochs[i].shape[0] - sliced_before.shape[0])

        if len(crossings) == 0:
            leftzero = sliced_before['Index'].loc[sliced_before['Signal'] == sliced_before['Signal'].min()]
            leftzero = np.array(leftzero)
            rightzero = sliced_after['Index'].loc[sliced_after['Signal'] == sliced_after['Signal'].min()]
            rightzero = np.array(rightzero)

        # upstroke and downstroker markers
        upstroke_idx = list(np.arange(leftzero, max_frame))
        upstroke = epochs[i].loc[epochs[i]['Index'].isin(upstroke_idx)]
        downstroke_idx = list(np.arange(max_frame, rightzero))
        downstroke = epochs[i].loc[epochs[i]['Index'].isin(downstroke_idx)]

        # left base and right base markers
        leftbase_idx = list(np.arange(epochs[i]['Index'].iloc[0], leftzero))
        leftbase_signal = epochs[i].loc[epochs[i]['Index'].isin(leftbase_idx)]
        leftbase_min = leftbase_signal['Signal'].min()
        leftbase = np.array(leftbase_signal['Index'].loc[leftbase_signal['Signal'] == leftbase_min])[0]

        rightbase_idx = list(np.arange(rightzero, epochs[i]['Index'].iloc[epochs[i].shape[0]-1]))
        rightbase_signal = epochs[i].loc[epochs[i]['Index'].isin(rightbase_idx)]
        rightbase_min = rightbase_signal['Signal'].min()
        rightbase = np.array(rightbase_signal['Index'].loc[rightbase_signal['Signal'] == rightbase_min])[0]

        # Rejecting candidate signals with low SNR (BAR = blink-amplitude-ratio)
        inside_blink_idx = list(np.arange(leftzero, rightzero))
        inside_blink = epochs[i].loc[epochs[i]['Index'].isin(inside_blink_idx)]
        outside_blink = pd.concat([leftbase_signal, rightbase_signal], axis=0)

        BAR = inside_blink.Signal.mean() / outside_blink.Signal[outside_blink['Signal'] > 0].mean()

        # BAR values in the range [5, 20] usually capture blinks reasonably well
        if not 3 < BAR < 50:
            candidate_blinks.append(epochs[i])
            peaks.append(max_frame)

        # Blink peak markers
        peaks = np.array(peaks)

    return candidate_blinks, peaks



#def _eog_blinks_distinguish(candidates):
#   """Distinguishing blinks from other eye movements
#   """




#
#
#
#
#    # 1Hz high pass is often helpful for fitting ICA
#    raw.filter(1., 40., n_jobs=2, fir_design='firwin')
#    n_max_eog = 3  # use max 3 components
#    eog_epochs = create_eog_epochs(raw, tmin=-.5, tmax=.5)
#    eog_epochs.decimate(5).apply_baseline((None, None))
#    eog_inds, scores_eog = ica.find_bads_eog(eog_epochs)
#    print('Found %d EOG component(s)' % (len(eog_inds),))
#    ica.exclude += eog_inds[:n_max_eog]
#    ica.plot_scores(scores_eog, exclude=eog_inds, title='EOG scores')
#
#
#

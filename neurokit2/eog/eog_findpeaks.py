# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..epochs import epochs_create, epochs_to_array
from ..misc import as_vector
from ..stats import fit_rmse, rescale
from ..signal import signal_findpeaks, signal_fixpeaks, signal_zerocrossings
from .eog_simulate import _eog_simulate_blink


def eog_findpeaks(eog_cleaned, sampling_rate=None, method="mne", **kwargs):
    """Locate EOG events (blinks, saccades, eye-movements, ...).

    Locate EOG events (blinks, saccades, eye-movements, ...).

    Parameters
    ----------
    eog_cleaned : Union[list, np.array, pd.Series]
        The cleaned EOG channel. Note that it must be positively oriented, i.e., blinks must
        appear as upward peaks.
    sampling_rate : int
        The signal sampling rate (in Hz, i.e., samples/second). Needed for method 'blinker' or
        'jammes2008'.
    method : str
        The peak detection algorithm. Can be one of 'mne' (default) (requires the MNE package
        to be installed), or 'brainstorm', 'blinker' or 'jammes2008'.
    sampling_rate : int
        The sampling frequency of the EOG signal (in Hz, i.e., samples/second). Needs to be supplied if the
        method to be used is 'blinker', otherwise defaults to None.
    **kwargs
        Other arguments passed to functions.

    Returns
    -------
    array
        Vector containing the samples at which EOG-peaks occur,

    See Also
    --------
    eog_clean

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Get data
    >>> eog_signal = nk.data('eog_100hz')
    >>> eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=100)
    >>>
    >>> # NeuroKit method
    >>> neurokit = nk.eog_findpeaks(eog_cleaned,
                                    sampling_rate=100,
                                    method="neurokit",
                                    threshold=0.4,
                                    show=True)
    >>> fig1 = nk.events_plot(neurokit, eog_cleaned)  # doctest: +ELLIPSIS
    >>> fig1
    >>>
    >>> # MNE-method
    >>> mne = nk.eog_findpeaks(eog_cleaned, method="mne")
    >>> fig2 = nk.events_plot(mne, eog_cleaned)  # doctest: +ELLIPSIS
    >>> fig2
    >>>
    >>> # brainstorm method
    >>> brainstorm = nk.eog_findpeaks(eog_cleaned, method="brainstorm")
    >>> fig3 = nk.events_plot(brainstorm, eog_cleaned)  # doctest: +ELLIPSIS
    >>> fig3
    >>>
    >>> # blinker method
    >>> blinker = nk.eog_findpeaks(eog_cleaned, sampling_rate=100, method="blinker")
    >>> fig4 = nk.events_plot(blinker, eog_cleaned)  # doctest: +ELLIPSIS
    >>> fig4
    >>>
    >>> # Jammes (2008) method
    >>> # jammes2008 = nk.eog_findpeaks(eog_cleaned, sampling_rate=100, method="jammes2008")
    >>> # fig5 = nk.events_plot(jammes2008, eog_cleaned)  # doctest: +ELLIPSIS
    >>> # fig5


    References
    ----------
    - Agarwal, M., & Sivakumar, R. (2019). Blink: A Fully Automated Unsupervised Algorithm for
    Eye-Blink Detection in EEG Signals. In 2019 57th Annual Allerton Conference on Communication,
    Control, and Computing (Allerton) (pp. 1113-1121). IEEE.
    - Kleifges, K., Bigdely-Shamlo, N., Kerick, S. E., & Robbins, K. A. (2017). BLINKER: automated
    extraction of ocular indices from EEG enabling large-scale analysis. Frontiers in neuroscience, 11, 12.

    """
    # Sanitize input
    eog_cleaned = as_vector(eog_cleaned)

    # Apply method
    method = method.lower()
    if method in ["mne"]:
        peaks = _eog_findpeaks_mne(eog_cleaned)
    elif method in ["brainstorm"]:
        peaks = _eog_findpeaks_brainstorm(eog_cleaned)
    elif method in ["blinker"]:
        peaks = _eog_findpeaks_blinker(eog_cleaned, sampling_rate=sampling_rate)
    elif method in ["neurokit", "nk"]:
        peaks = _eog_findpeaks_neurokit(eog_cleaned, sampling_rate=sampling_rate, **kwargs)
#    elif method in ["jammes2008", "jammes"]:
#        peaks = _eog_findpeaks_jammes2008(eog_cleaned, sampling_rate=sampling_rate)
    else:
        raise ValueError("NeuroKit error: eog_peaks(): 'method' should be " "one of 'mne', 'brainstorm' or 'blinker'.")

    return peaks


# =============================================================================
# Methods
# =============================================================================
def _eog_findpeaks_neurokit(eog_cleaned, sampling_rate=1000, threshold=0.5, show=True):
    """In-house EOG blink detection.

    """
    peaks = signal_findpeaks(eog_cleaned, relative_height_min=1)["Peaks"]
    peaks = signal_fixpeaks(peaks=peaks,
                            sampling_rate=sampling_rate,
                            interval_min=0.25,
                            method="neurokit")
    peaks = _eog_findpeaks_neurokit_filterblinks(eog_cleaned,
                                                 peaks,
                                                 sampling_rate=sampling_rate,
                                                 threshold=threshold,
                                                 show=show)
    return peaks


def _eog_findpeaks_neurokit_filterblinks(eog_cleaned, peaks, sampling_rate=1000, threshold=0.5, show=False):
    """Compare each detected event to blink template and reject it if too different.
    """
    # Get epoch around each blink
    events = epochs_create(eog_cleaned,
                              peaks,
                              sampling_rate=sampling_rate,
                              epochs_start=-0.4,
                              epochs_end=0.6)
    events = epochs_to_array(events)  # Convert to 2D array

    # Generate Blink-template
    template = _eog_simulate_blink(sampling_rate=sampling_rate)

    # Get the "distance" (RMSE) between each blink and the template
    rmse = np.full(events.shape[1], np.nan)
    for i in range(events.shape[1]):
        events[:, i] = rescale(events[:, i], to=[0, 1])  # Reshape to 0-1 scale
        rmse[i] = fit_rmse(events[:, i], template)

    # Plot RMSE distribution
    if show is True:
        plt.subplot(1, 2, 1)
        plt.hist(rmse, color = '#FF9800')
        plt.axvline(x=threshold, linewidth=4, color='r')
        plt.title('RMSE Distribution (threshold = '+str(threshold) + ')')
        plt.xlabel('RMSE')

        plt.subplot(1, 2, 2)
        plt.plot(events[:, rmse < threshold], linewidth=0.25, color="black")
        plt.plot(events[:, rmse >= threshold], linewidth=0.5, color="red")
        plt.plot(template, linewidth=2, color="#2196F3", label='Blink template')
        plt.title('Accepted and rejected (red) blinks')
        plt.legend(loc="upper right")

    return peaks[rmse < threshold]





#def _eog_findpeaks_jammes2008(eog_cleaned, sampling_rate=1000):
#    """Derivative-based method by Jammes (2008)
#
#    https://link.springer.com/article/10.1007/s11818-008-0351-y
#
#    """
#    # Derivative
#    derivative = np.gradient(eog_cleaned)
#
#    # These parameters were set by the authors "empirically". These are values based on
#    # their figure 1.
#    vcl = 0.5 * np.max(derivative)
#    vol = 0.75 * np.min(derivative)
#
#    crosses_vcl = signal_zerocrossings(derivative - vcl, direction="up")
#    crosses_vol = signal_zerocrossings(derivative - vol, direction="down")
#    crosses_vol = nk.find_closest(crosses_vcl, crosses_vol, direction="above")
#
#    nk.events_plot([crosses_vcl, crosses_vol], eog_cleaned)
#    nk.signal_plot([eog_cleaned, derivative, derivative - vol])
#    durations = (crosses_vol - crosses_vcl) / sampling_rate
#    indices = durations < 0.5
#
#    peaks = np.full(np.sum(indices), np.nan)
#    for i in range(np.sum(indices)):
#        segment = eog_cleaned[crosses_vcl[indices][i]:crosses_vol[indices][i]]
#        peaks[i] = crosses_vcl[indices][i] + np.argmax(segment)
#
#    return peaks


def _eog_findpeaks_mne(eog_cleaned):
    """EOG blink detection based on MNE.

    https://github.com/mne-tools/mne-python/blob/master/mne/preprocessing/eog.py

    """
    # Make sure MNE is installed
    try:
        import mne
    except ImportError:
        raise ImportError(
            "NeuroKit error: signal_filter(): the 'mne' module is required for this method to run. ",
            "Please install it first (`pip install mne`).",
        )

    # Find peaks
    eog_events, _ = mne.preprocessing.peak_finder(eog_cleaned, extrema=1, verbose=False)

    return eog_events


def _eog_findpeaks_brainstorm(eog_cleaned):
    """EOG blink detection implemented in brainstorm.

    https://neuroimage.usc.edu/brainstorm/Tutorials/ArtifactsDetect#Detection:_Blinks

    """
    # Brainstorm: "An event of interest is detected if the absolute value of the filtered
    # signal value goes over a given number of times the standard deviation. For EOG: 2xStd."
    # -> Remove all peaks that correspond to regions < 2 SD
    peaks = signal_findpeaks(eog_cleaned, relative_height_min=2)["Peaks"]

    return peaks


def _eog_findpeaks_blinker(eog_cleaned, sampling_rate=1000):
    """EOG blink detection based on BLINKER algorithm.

    Detects only potential blink landmarks and does not separate blinks from other artifacts yet.
    https://www.frontiersin.org/articles/10.3389/fnins.2017.00012/full

    """
    # Establish criterion
    threshold = 1.5 * np.std(eog_cleaned) + eog_cleaned.mean()
    min_blink = 0.05 * sampling_rate  # min blink frames

    potential_blinks = []
    for i in range(len(eog_cleaned)):
        if eog_cleaned[i] > threshold:
            potential_blinks.append(i)

    # Make sure each blink is 50ms long and separated by 50ms
    indexes = np.where(np.diff(potential_blinks) > min_blink)[0]
    individual_blinks = np.split(np.diff(potential_blinks), indexes)

    blinks = []
    for idx, i in enumerate(individual_blinks):
        if len(i) > min_blink:
            blinks.append(idx)

    candidates = np.array(potential_blinks)[np.append(0, indexes)[blinks]]

    # Calculate blink landmarks
    epochs = epochs_create(
        eog_cleaned, events=candidates, sampling_rate=sampling_rate, epochs_start=-0.5, epochs_end=0.5
    )

    # max value marker
    candidate_blinks = []
    peaks = []
    for i in epochs:
        max_value = epochs[i].Signal.max()

        # Check if peak is at the end or start of epoch
        t = epochs[i].loc[epochs[i]["Signal"] == max_value].index
        if np.all(0.3 < t < 0.51):
            # Trim end of epoch
            epochs[i] = epochs[i][-0.5:0.3]
            max_value = epochs[i].Signal.max()
        if np.all(-0.51 < t < -0.3):
            # Trim start of epoch
            epochs[i] = epochs[i][-0.3:0.5]
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
            leftzero = np.array(leftzero)

        if (max_position + 1) < len(crossings_idx):  # crosses zero point
            rightzero = crossings_idx[max_position + 1]
        else:
            max_value_t = epochs[i].Signal.idxmax()
            sliced_before = epochs[i].loc[slice(max_value_t), :]
            sliced_after = epochs[i].tail(epochs[i].shape[0] - sliced_before.shape[0])
            rightzero = sliced_after["Index"].loc[sliced_after["Signal"] == sliced_after["Signal"].min()]
            rightzero = np.array(rightzero)

        # upstroke and downstroke markers
        #        upstroke_idx = list(np.arange(leftzero, max_frame))
        #        upstroke = epochs[i].loc[epochs[i]['Index'].isin(upstroke_idx)]
        #        downstroke_idx = list(np.arange(max_frame, rightzero))
        #        downstroke = epochs[i].loc[epochs[i]['Index'].isin(downstroke_idx)]

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

        # BAR values in the range [5, 20] usually capture blinks reasonably well
        if not 3 < BAR < 50:
            candidate_blinks.append(epochs[i])
            peaks.append(max_frame)

    # Blink peak markers
    peaks = np.array(peaks)

    return peaks

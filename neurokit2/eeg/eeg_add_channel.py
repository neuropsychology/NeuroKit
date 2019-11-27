# -*- coding: utf-8 -*-
import mne

import pandas as pd
import numpy as np





def eeg_add_channel(raw, channel, channel_type=None, channel_name=None, sync_index_raw=0, sync_index_channel=0):
    """
    Add a channel to a mne's Raw m/eeg file. It will basically synchronize the channel to the eeg data following a particular index and add it.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    channel : list or array
        The channel to be added.
    channel_type : str
        Channel type. Currently supported fields are 'ecg', 'bio', 'stim', 'eog', 'misc', 'seeg', 'ecog', 'mag', 'eeg', 'ref_meg', 'grad', 'emg', 'hbr' or 'hbo'.
    channel_type : str
        Channel name.
    sync_index_raw, sync_index_channel : int or list
        An index, in the raw data and in the channel to add, by which to align the two inputs.

    Returns
    ----------
    mne.io.Raw
        Raw data in FIF format.

    Example
    ----------
    >>> import neurokit as nk
    >>> event_index_in_eeg = 42
    >>> event_index_in_ecg = 333
    >>> raw = nk.eeg_add_channel(raw, ecg, sync_index_raw=event_index_in_eeg, sync_index_channel=event_index_in_ecg, channel_type="ecg")
    """
    if channel_name is None:
        if isinstance(channel, pd.Series):
            if channel.name is not None:
                channel_name = channel.name
            else:
                channel_name = "Added_Channel"
        else:
            channel_name = "Added_Channel"

    # Compute the distance between the two signals
    diff = sync_index_channel - sync_index_raw
    if diff > 0:
        channel = list(channel)[diff:len(channel)]
        channel = channel + [np.nan]*diff
    if diff < 0:
        channel = [np.nan]*diff + list(channel)
        channel = list(channel)[0:len(channel)]

    # Adjust to raw size
    if len(channel) < len(raw):
        channel = list(channel) + [np.nan]*(len(raw)-len(channel))
    else:
        channel = list(channel)[0:len(raw)]  # Crop to fit the raw data

    info = mne.create_info([channel_name], raw.info["sfreq"], ch_types=channel_type)
    channel = mne.io.RawArray([channel], info)

    raw.add_channels([channel], force_update_info=True)

    return(raw)



# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def mne_channel_add(raw, channel, channel_type=None, channel_name=None, sync_index_raw=0, sync_index_channel=0):
    """
    Add channel as array to MNE.

    Add a channel to a mne's Raw m/eeg file. It will basically synchronize the channel to the eeg data following a particular index and add it.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data from MNE.
    channel : list or array
        The signal to be added.
    channel_type : str
        Channel type. Currently supported fields are 'ecg', 'bio', 'stim', 'eog', 'misc', 'seeg', 'ecog', 'mag', 'eeg', 'ref_meg', 'grad', 'emg', 'hbr' or 'hbo'.
    channel_type : str
        Desired channel name.
    sync_index_raw, sync_index_channel : int or list
        An index (e.g., the onset of the same event marked in the same signal), in the raw data and in the channel to add, by which to align the two inputs. This can be used in case the EEG data and the channel to add do not have the same onsets and must be aligned through some common event.

    Returns
    ----------
    mne.io.Raw
        Raw data in FIF format.

    Example
    ----------
    >>> import neurokit2 as nk
    >>> import mne
    >>>
    >>> # Let's say that the 42nd sample point in the EEG correspond to the 333rd point in the ECG
    >>> event_index_in_eeg = 42
    >>> event_index_in_ecg = 333
    >>>
    >>> raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_raw.fif', preload=True) # doctest: +SKIP
    >>> ecg = nk.ecg_simulate(length=170000)
    >>>
    >>> raw = nk.mne_channel_add(raw, ecg, sync_index_raw=event_index_in_eeg, sync_index_channel=event_index_in_ecg, channel_type="ecg") # doctest: +SKIP

    """
    # Try loading mne
    try:
        import mne
    except ImportError:
        raise ImportError(
            "NeuroKit error: eeg_add_channel(): the 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        )

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
        channel = list(channel)[diff : len(channel)]
        channel = channel + [np.nan] * diff
    if diff < 0:
        channel = [np.nan] * abs(diff) + list(channel)

    # Adjust to raw size
    if len(channel) < len(raw):
        channel = list(channel) + [np.nan] * (len(raw) - len(channel))
    else:
        # Crop to fit the raw data
        channel = list(channel)[0 : len(raw)]

    info = mne.create_info([channel_name], raw.info["sfreq"], ch_types=channel_type)
    channel = mne.io.RawArray([channel], info)

    raw.add_channels([channel], force_update_info=True)

    return raw

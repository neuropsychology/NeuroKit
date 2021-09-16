# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def mne_channel_extract(raw, what, name=None, add_firstsamples=False):
    """Channel array extraction from MNE.

    Select one or several channels by name and returns them in a dataframe.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    what : str or list
        Can be 'MEG', which will extract all MEG channels, 'EEG', which will extract all EEG channels, or 'EOG',
        which will extract all EOG channels (that is, if channel names are named with prefixes of their type e.g.,
        'EEG 001' etc. or 'EOG 061'). Provide exact a single or a list of channel's name(s) if not
        (e.g., ['124', '125']).
    name : str or list
        Useful only when extracting one channel. Can also take a list of names for renaming multiple channels,
        Otherwise, defaults to None.
    add_firstsamples : bool
        Defaults to False. Returns first few samples of NaN rows of data corresponding to the offset between
        the start of the system and the start of the recording if True.        
        See https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.first_samp.

    Returns
    ----------
    DataFrame
        A DataFrame or Series containing the channel(s).

    Example
    ----------
    >>> import neurokit2 as nk
    >>> import mne
    >>>
    >>> raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() +
    ...                           '/MEG/sample/sample_audvis_raw.fif', preload=True) #doctest: +SKIP
    >>>
    >>> raw_channel = nk.mne_channel_extract(raw, what=["EEG 060", "EEG 055"], name=['060', '055']) # doctest: +SKIP
    >>> eeg_channels = nk.mne_channel_extract(raw, "EEG") # doctest: +SKIP
    >>> eog_channels = nk.mne_channel_extract(raw, what='EOG', name='EOG') # doctest: +SKIP

    """
    channels_all = raw.copy().info["ch_names"]

    # Select category of channels
    if what in ["EEG", "EOG", "MEG"]:
        what = [x for x in channels_all if what in x]
    # Select a single specified channel
    elif isinstance(what, str):
        what = [what]
    # Select a few specified channels
    elif isinstance(what, list):
        if not all(x in channels_all for x in what):
            raise ValueError(
                "NeuroKit error: mne_channel_extract(): List of channels not found. Please "
                "check channel names in raw.info['ch_names']. "
            )

    channels, __ = raw.copy().pick_channels(what)[:]
    if len(what) > 1:
        channels = pd.DataFrame(channels.T, columns=what)
        if name is not None:
            channels.columns = name
    else:
        channels = pd.Series(channels[0])
        channels.what = what[0]
        if name is not None:
            channels = channels.rename(name)
    
    # Add first_samp
    if add_firstsamples:

        add_na = np.repeat(np.nan, raw.first_samp)
        if isinstance(channels, pd.Series):
            new_channels = pd.concat([pd.Series(add_na), channels], axis=0)
            new_channels.name = channels.name

        elif isinstance(channels, pd.DataFrame):
            new_channels = pd.DataFrame()
            for name in channels.columns.values:
                new_channels[name] = pd.concat([pd.Series(add_na), channels[name]], axis=0)

        channels = new_channels.copy()

    return channels

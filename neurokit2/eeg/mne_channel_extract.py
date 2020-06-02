# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def mne_channel_extract(raw, name):
    """
    Channel array extraction from MNE.

    Select one or several channels by name and returns them in a dataframe.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    channel_names : str or list
        Channel's name(s).

    Returns
    ----------
    DataFrame
        A DataFrame or Series containing the channel(s).

    Example
    ----------
    >>> import neurokit2 as nk
    >>> import mne
    >>>
    >>> raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_raw.fif', preload=True) #doctest: +SKIP
    >>>
    >>> raw_channel = nk.mne_channel_extract(raw, "EEG 055") # doctest: +SKIP

    """
    if isinstance(name, list) is False:
        name = [name]

    channels, time_index = raw.copy().pick_channels(name)[:]
    if len(name) > 1:
        channels = pd.DataFrame(channels.T, columns=name)
    else:
        channels = pd.Series(channels[0])
        channels.name = name[0]
    return channels

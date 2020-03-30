# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np





def mne_channel_extract(raw, name):
    """Channel array extraction from MNE

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
    >>> raw = nk.mne_channel_extract(raw, "TP7")
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

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def mne_channel_extract(raw, what, name=None, add_firstsamples=False):
    """**Channel extraction from MNE objects**

    Select one or several channels by name and returns them in a dataframe.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    what : str or list
        Can be ``"MEG"``, which will extract all MEG channels, ``"EEG"``, which will extract all EEG
        channels, or ``"EOG"``, which will extract all EOG channels (that is, if channel names are
        named with prefixes of their type e.g., 'EEG 001' etc. or 'EOG 061'). Provide exact a single
        or a list of channel's name(s) if not (e.g., ['124', '125']).
    name : str or list
        Useful only when extracting one channel. Can also take a list of names for renaming multiple channels,
        Otherwise, defaults to ``None``.
    add_firstsamples : bool
        Defaults to ``False``. MNE's objects store the value of a delay between
        the start of the system and the start of the recording
        (see https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.first_samp).
        Taking this into account can be useful when extracting channels from the Raw object to
        detect events indices that are passed back to MNE again. When ``add_firstsamples`` is set to
        ``True``, the offset will be explicitly added at the beginning of the signal and filled with
        NaNs. If ``add_firstsamples`` is a float or an integer, the offset will filled with these
        values instead. If it is set to ``backfill``, will prepend with the first real value.

    Returns
    ----------
    DataFrame
        A DataFrame or Series containing the channel(s).

    Example
    ----------
    .. ipython:: python

      import neurokit2 as nk
      import mne

      raw = nk.mne_data("raw")

      raw_channel = nk.mne_channel_extract(raw, what=["EEG 060", "EEG 055"], name=['060', '055'])
      eeg_channels = nk.mne_channel_extract(raw, "EEG")

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

    channels, __ = raw.copy().pick_channels(what, ordered=False)[:]
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
    if isinstance(add_firstsamples, bool) and add_firstsamples is True:  # Fill with na
        add_firstsamples = np.nan
    if isinstance(add_firstsamples, str):  # Back fill
        add_firstsamples = channels.iloc[0]
        if isinstance(channels, pd.DataFrame):
            add_firstsamples = dict(add_firstsamples)

    if add_firstsamples is not False:
        if isinstance(channels, pd.Series):
            fill = pd.Series(add_firstsamples, index=range(-raw.first_samp, 0))
            channels = pd.concat([fill, channels], axis=0)
        elif isinstance(channels, pd.DataFrame):
            fill = pd.DataFrame(
                add_firstsamples, index=range(-raw.first_samp, 0), columns=channels.columns
            )
            channels = pd.concat([fill, channels], axis=0)

    return channels

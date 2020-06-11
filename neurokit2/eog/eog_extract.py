# -*- coding: utf-8 -*-
import numpy as np

from ..eeg import mne_channel_add
from ..eeg import mne_channel_extract


def eog_extract(raw, channels, resampling_rate=None, raw_return=False, show=False):
    """Extract EOG signal from the EEG data. Wrapper for MNE.

    Two EOG channels are used to monitor both horizontal and vertical eye movements. Usually, electrodes are
    placed at the right and left outer canthi, one above and one below the horizontal eye axis. The
    electrodes pick up the inherent voltage within the eye; the cornea has a positive charge and the retina
    has a negative charge.


    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    channels : list
        List of EOG channel names.
    resampling_rate : int
        The sampling rate (in Hz, i.e., samples/second) at which to resample EEG data. Defaults to None.
    raw_return : bool
        Adds EOG channel to EEG data (RawEDF object) if True.
    show : bool
        Returns plot of EOG signal if True.

    Returns
    -------
    eog : Series
        A Series containing the EOG signal.

    See Also
    --------
    events_find, mne_channel_add, mne_channel_extract

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> eog = nk.eog_extract(raw, channels=["124", "125"], resampling_rate=None, raw_return=False)
    >>> eog = nk.eog_extract(raw, channels=["124", "125"], resampling_rate=None, raw_return=True)

    """
    # Sanity checks
    if not isinstance(raw, mne.io.BaseRaw):
        raise ValueError("NeuroKit warning: eog_extract(): Please make sure your EEG data is an mne.io.Raw object.")

    # Load eeg data
    if resampling_rate is not None:
        raw = raw.resample(resampling_rate, npad='auto')

    # Extract EOG channels
    if len(channels) != 2:
        raise ValueError("NeuroKit warning: eog_extract(): Please make sure your channels contain exactly 2 EOG names.")

    eog = mne_channel_extract(raw, name=[channels[0], channels[1]])
    eog = eog.iloc[:, 0] - eog.iloc[:, 1]

    # Add EOG channel to EEG data (raw)
    if raw_return:
        raw = mne_channel_add(raw, eog, channel_type="eog", channel_name="EOG")
        raw = raw.drop_channels([channels[0], channels[1]])

    if show:
        eog.plot()

    return eog

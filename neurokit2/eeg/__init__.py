"""Submodule for NeuroKit."""

from .mne_data import mne_data
from .mne_channel_add import mne_channel_add
from .mne_channel_extract import mne_channel_extract
from .mne_to_df import mne_to_df, mne_to_dict
from .eeg_rereference import eeg_rereference
from .eeg_gfp import eeg_gfp
from .eeg_diss import eeg_diss
from .eeg_badchannels import eeg_badchannels


__all__ = ["mne_data",
           "mne_channel_add",
           "mne_channel_extract",
           "mne_to_df",
           "mne_to_dict",
           "eeg_rereference",
           "eeg_gfp",
           "eeg_diss",
           "eeg_badchannels"]

"""Submodule for NeuroKit."""

from .eeg_badchannels import eeg_badchannels
from .eeg_diss import eeg_diss
from .eeg_gfp import eeg_gfp
from .eeg_power import eeg_power
from .eeg_rereference import eeg_rereference
from .eeg_simulate import eeg_simulate
from .eeg_source import eeg_source
from .eeg_source_extract import eeg_source_extract
from .mne_channel_add import mne_channel_add
from .mne_channel_extract import mne_channel_extract
from .mne_crop import mne_crop
from .mne_data import mne_data
from .mne_templateMRI import mne_templateMRI
from .mne_to_df import mne_to_df, mne_to_dict

__all__ = [
    "mne_data",
    "mne_channel_add",
    "mne_channel_extract",
    "mne_crop",
    "mne_to_df",
    "mne_to_dict",
    "mne_templateMRI",
    "eeg_simulate",
    "eeg_source",
    "eeg_source_extract",
    "eeg_power",
    "eeg_rereference",
    "eeg_gfp",
    "eeg_diss",
    "eeg_badchannels",
]

# -*- coding: utf-8 -*-
import os
import mne


def example_data_eeg():
    """Download and load an EEG example dataset.
    """
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif')
    raw = mne.io.read_raw_fif(sample_data_raw_file)
    return(raw)
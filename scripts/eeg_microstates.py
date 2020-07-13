import mne
import numpy as np
import pandas as pd
import mne_microstates
import neurokit2 as nk
import matplotlib.pyplot as plt

# Read original file (too big to be uploaded on github)
#raw = mne.io.read_raw_fif("../data/eeg_restingstate_300hz.fif", preload=True)
raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw.fif', preload=True)
events = mne.read_events(mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif')

# Create epochs including different events
event_id = {'audio/left': 1, 'audio/right': 2,
            'visual/left': 3, 'visual/right': 4}


# Selecting the sensor types to use in the analysis. In this example, we
# use only EEG channels
raw = raw.pick_types(meg=False, eeg=True)

# Always use an average EEG reference when doing microstate analysis
raw = raw.set_eeg_reference('average')

# Highpass filter the data a little bit
raw = raw.filter(0.2, None)

# Segment the data into 6 microstates
topos, microstates = mne_microstates.segment(raw.get_data(), n_states=6)

# Plot the topographic maps of the found microstates
#mne_microstates.plot_maps(topos, raw.info)

# Plot the segmentation of the first 500 samples
#mne_microstates.plot_segmentation(microstates[:500], raw.get_data()[:, :500], raw.times[:500])


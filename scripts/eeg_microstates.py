import mne
import numpy as np
import pandas as pd

import neurokit2 as nk

# Read original file (too big to be uploaded on github)
raw = mne.io.read_raw_fif("../data/eeg_restingstate_300hz.fif", preload=True)

# Always use an average EEG reference when doing microstate analysis
raw.set_eeg_reference('average')

# Highpass filter the data a little bit
raw.filter(0.2, None)

# Selecting the sensor types to use in the analysis. In this example, we
# use only EEG channels
raw.pick_types(meg=False, eeg=True)

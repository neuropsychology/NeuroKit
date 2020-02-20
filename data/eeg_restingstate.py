#import TruScanEEGpy
import mne

# Read original file (too big to be uploaded on github)
raw = mne.io.read_raw_edf("eeg_restingstate_3000hz.edf", preload=True)
raw = raw.resample(300)

# Add montage here

# Save
raw.save("eeg_restingstate_300hz.fif", overwrite=True)

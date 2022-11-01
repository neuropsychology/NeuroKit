import mne
import numpy as np
import TruScanEEGpy

import neurokit2 as nk

# EDF TO FIF
# ==========
# Read original file (too big to be uploaded on github)
raw = mne.io.read_raw_edf("eeg_restingstate_3000hz.edf", preload=True)

# Find event onset and cut
event = nk.events_find(raw.copy().pick_channels(["Foto"]).to_data_frame()["Foto"])
tmin = event["onset"][0] / 3000
raw = raw.crop(tmin=tmin, tmax=tmin + 8 * 60)


# EOG
eog = raw.copy().pick_channels(["124", "125"]).to_data_frame()
eog = eog["124"] - eog["125"]
raw = nk.eeg_add_channel(raw, eog, channel_type="eog", channel_name="EOG")
raw = raw.drop_channels(["124", "125"])


# Montage
mne.rename_channels(
    raw.info, dict(zip(raw.info["ch_names"], TruScanEEGpy.convert_to_tenfive(raw.info["ch_names"])))
)
montage = TruScanEEGpy.montage_mne_128(TruScanEEGpy.layout_128(names="10-5"))
extra_channels = np.array(raw.info["ch_names"])[
    np.array([i not in montage.ch_names for i in raw.info["ch_names"]])
]
raw = raw.drop_channels(extra_channels[np.array([i not in ["EOG"] for i in extra_channels])])
raw = raw.set_montage(montage)


# Save
raw = raw.resample(300)
raw.save("eeg_restingstate_300hz.fif", overwrite=True)


## Convert to df
# df = pd.DataFrame(raw.get_data().T)
# df.columns = raw.info["ch_names"]
# df.to_csv("eeg_restingstate_300hz.csv")

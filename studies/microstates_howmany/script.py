import os
import mne
import scipy
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import autoreject
from autoreject.utils import interpolate_bads
import scipy.stats


data_path = "D:/Dropbox/RECHERCHE/N/NeuroKit/data/rs_eeg_texas/data/"
files = os.listdir(data_path)


results = []
for i, file in enumerate(files[0:2]):
    print(i)

    # Read
    raw = mne.io.read_raw_bdf(data_path + file, eog=['LVEOG', 'RVEOG', 'LHEOG', 'RHEOG'], misc=['M1', 'M2', 'NAS', 'NFpz'], preload=True)
    sampling_rate = np.rint(raw.info["sfreq"])

    # Set montage
    raw = raw.set_montage("biosemi64")

    # Find events
    events = nk.events_find(nk.mne_channel_extract(raw, "Status"),
                            threshold_keep="below",
                            event_conditions=["EyesClosed", "EyesOpen"] * 4)

    # Rereference
    raw = nk.eeg_rereference(raw, ["M1", "M2"])

    # Filter
    raw = raw.filter(1, 35)

    # Bad epochs
    bads, info = nk.eeg_badchannels(raw)
    raw.info['bads'] += bads
#    raw.plot()
    raw = raw.interpolate_bads()

    # ICA
    ica = mne.preprocessing.ICA(n_components=15, random_state=97).fit(raw)
    ica = ica.detect_artifacts(raw, eog_ch=['LVEOG', 'RVEOG'])
#    ica.plot_properties(raw, picks=ica.exclude)
    raw = ica.apply(raw)

    # Rereference
    raw = nk.eeg_rereference(raw, "average")


    for method in ["kmdo"]:
        rez = nk.microstates_findnumber(raw, n_max=6, show=False, method="kmod")
        rez["Method"] = method
        rez["Participant"] = file
        results.append(rez)



#
#
#
#
#
#    ransac = autoreject.Ransac(verbose='progressbar', picks="eeg", n_jobs=1)
#    raw = autoreject.get_rejection_threshold(raw, picks="eeg")
#
#
#raw.info["ch_names"]
#nk.signal_plot(event)
#mne.viz.plot_raw(raw)




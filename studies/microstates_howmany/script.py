import os
import mne
import scipy
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import autoreject
from autoreject.utils import interpolate_bads  # noqa
import scipy.stats


data_path = "D:/Dropbox/RECHERCHE/N/NeuroKit/data/rs_eeg_texas/data/"
files = os.listdir(data_path)

for i, file in enumerate(files):
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
    raw = nk.eeg_rereference(raw, "average")

    # Filter
    raw = raw.filter(1, 35)

    # ICA
    ica = mne.preprocessing.ICA(n_components=15, random_state=97).fit(raw)
    ica.plot_properties(raw, picks=[0, 1])

    ica = ica.detect_artifacts(raw, eog_ch=['LVEOG', 'RVEOG'])
    raw = ica.apply(raw)


    ransac = autoreject.Ransac(verbose='progressbar', picks="eeg", n_jobs=1)
    raw = autoreject.get_rejection_threshold(raw, picks="eeg")





raw.info["ch_names"]
nk.signal_plot(event)
mne.viz.plot_raw(raw)




def eeg_badchannels(eeg):
    """Find bad channels
    """
    data, time = raw[mne.pick_types(raw.info, eeg=True)]

    results = []
    for i in range(len(data)):
        channel = data[i, :]
        info = {"Channel": [i],
                "SD": [np.nanstd(channel, ddof=1)],
                "Mean": [np.nanmean(channel)]}
        results.append(pd.DataFrame(info))
    results = pd.concat(results, axis=0)
    results = results.set_index("Channel")

    z = nk.standardize(results)
    results["Outlier"] = (z.abs() > scipy.stats.norm.ppf(.99)).sum(axis=1) / len(results.columns)
    bads = np.where(results["Outlier"] >= 0.5)[0]


# -*- coding: utf-8 -*-
"""Script for formatting the Fantasia Database

The database consists of twenty young and twenty elderly healthy subjects. All subjects remained in a resting state in sinus rhythm while watching the movie Fantasia (Disney, 1940) to help maintain wakefulness. The continuous ECG signals were digitized at 250 Hz. Each heartbeat was annotated using an automated arrhythmia detection algorithm, and each beat annotation was verified by visual inspection.

Steps:
    1. Download the ZIP database from https://physionet.org/content/fantasia/1.0.0/
    2. Open it with a zip-opener (WinZip, 7zip).
    3. Extract the folder of the same name (named 'fantasia-database-1.0.0') to the same folder as this script.
    4. Run this script.
"""
import pandas as pd
import numpy as np
import wfdb
import os


files = os.listdir("./fantasia-database-1.0.0/")
files = [s.replace('.dat', '') for s in files if ".dat" in s]

dfs_ecg = []
dfs_rpeaks = []


for i, participant in enumerate(files):

    data, info = wfdb.rdsamp("./fantasia-database-1.0.0/" + participant)

    # Get signal
    data = pd.DataFrame(data, columns=info["sig_name"])
    data = data[["ECG"]]
    data["Participant"] = "Fantasia_" + participant
    data["Sample"] = range(len(data))
    data["Sampling_Rate"] = info['fs']
    data["Database"] = "Fantasia"

    # Get annotations
    anno = wfdb.rdann("./fantasia-database-1.0.0/" + participant, 'ecg')
    anno = anno.sample[np.where(np.array(anno.symbol) == "N")[0]]
    anno = pd.DataFrame({"Rpeaks": anno})
    anno["Participant"] = "Fantasia_" + participant
    anno["Sampling_Rate"] = info['fs']
    anno["Database"] = "Fantasia"

    # Store with the rest
    dfs_ecg.append(data)
    dfs_rpeaks.append(anno)


# Save
df_ecg = pd.concat(dfs_ecg).to_csv("ECGs.csv", index=False)
dfs_rpeaks = pd.concat(dfs_rpeaks).to_csv("Rpeaks.csv", index=False)

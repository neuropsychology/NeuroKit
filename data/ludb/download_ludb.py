# -*- coding: utf-8 -*-
"""Script for formatting the Lobachevsky University Electrocardiography Database

The database consists of 200 10-second 12-lead ECG signal records representing different morphologies of the ECG signal. The ECGs were collected from healthy volunteers and patients, which had various cardiovascular diseases. The boundaries of P, T waves and QRS complexes were manually annotated by cardiologists for all 200 records.

Steps:
    1. In the command line, run 'pip install gsutil'
    2. Then, 'gsutil -m cp -r gs://ludb-1.0.0.physionet.org D:/YOURPATH/NeuroKit/data/ludb'
    This will download all the files in a folder named 'ludb-1.0.0.physionet.org' at the
    destination you entered.
    3. Run this script.
"""
import pandas as pd
import numpy as np
import wfdb
import os


files = os.listdir("./ludb-1.0.0.physionet.org/")

dfs_ecg = []
dfs_rpeaks = []


for participant in range(200):
    filename = str(participant + 1)

    data, info = wfdb.rdsamp("./ludb-1.0.0.physionet.org/" + filename)

    # Get signal
    data = pd.DataFrame(data, columns=info["sig_name"])
    data = data[["i"]].rename(columns={"i": "ECG"})
    data["Participant"] = "LUDB_%.2i" %(participant + 1)
    data["Sample"] = range(len(data))
    data["Sampling_Rate"] = info['fs']
    data["Database"] = "LUDB"

    # Get annotations
    anno = wfdb.rdann("./ludb-1.0.0.physionet.org/" + filename, 'atr_i')
    anno = anno.sample[np.where(np.array(anno.symbol) == "N")[0]]
    anno = pd.DataFrame({"Rpeaks": anno})
    anno["Participant"] = "LUDB_%.2i" %(participant + 1)
    anno["Sampling_Rate"] = info['fs']
    anno["Database"] = "LUDB"

    # Store with the rest
    dfs_ecg.append(data)
    dfs_rpeaks.append(anno)


# Save
df_ecg = pd.concat(dfs_ecg).to_csv("ECGs.csv", index=False)
dfs_rpeaks = pd.concat(dfs_rpeaks).to_csv("Rpeaks.csv", index=False)

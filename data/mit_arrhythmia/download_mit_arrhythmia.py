# -*- coding: utf-8 -*-
"""Script for formatting the MIT-Arrhythmia database

Steps:
    1. Download the ZIP database from https://alpha.physionet.org/content/mitdb/1.0.0/
    2. Open it with a zip-opener (WinZip, 7zip).
    3. Extract the folder of the same name (named 'mit-bih-arrhythmia-database-1.0.0') to the same folder as this script.
    4. Run this script.

Credits:
    https://github.com/berndporr/py-ecg-detectors/blob/master/tester_MITDB.py by Bernd Porr
"""
import pandas as pd
import numpy as np
import wfdb
import os
import neurokit2 as nk

database_path = "./mit-bih-arrhythmia-database-1.0.0/"

# Check if expected folder exists
if not os.path.exists(database_path):
    url = "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
    download_successful = nk.download_zip(url, database_path)
    if not download_successful:
        raise ValueError(
            "NeuroKit error: download of MIT-Arrhythmia database failed. "
            "Please download it manually from https://alpha.physionet.org/content/mitdb/1.0.0/ "
            "and unzip it in the same folder as this script."
        )

data_files = [database_path + file for file in os.listdir(database_path) if ".dat" in file]

def read_file(file, participant):
    """Utility function
    """
    # Get signal
    data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4])[0][:, 0]})
    data["Participant"] = "MIT-Arrhythmia_%.2i" %(participant)
    data["Sample"] = range(len(data))
    data["Sampling_Rate"] = 360
    data["Database"] = "MIT-Arrhythmia-x" if "x_mitdb" in file else "MIT-Arrhythmia"

    # getting annotations
    anno = wfdb.rdann(file[:-4], 'atr')
    anno = np.unique(anno.sample[np.in1d(anno.symbol, ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'])])
    anno = pd.DataFrame({"Rpeaks": anno})
    anno["Participant"] = "MIT-Arrhythmia_%.2i" %(participant)
    anno["Sampling_Rate"] = 360
    anno["Database"] = "MIT-Arrhythmia-x" if "x_mitdb" in file else "MIT-Arrhythmia"

    return data, anno




dfs_ecg = []
dfs_rpeaks = []

for participant, file in enumerate(data_files):

    print("Participant: " + str(participant + 1) + "/" + str(len(data_files)))

    data, anno = read_file(file, participant)

    # Store with the rest
    dfs_ecg.append(data)
    dfs_rpeaks.append(anno)

    # Store additional recording if available
    if "x_" + file.replace(database_path, "") in os.listdir(database_path + "x_mitdb/"):
        print("  - Additional recording detected.")
        data, anno = read_file(database_path + "/x_mitdb/" + "x_" + file.replace(database_path, ""), participant)
        # Store with the rest
        dfs_ecg.append(data)
        dfs_rpeaks.append(anno)



# Save
df_ecg = pd.concat(dfs_ecg).to_csv("ECGs.csv", index=False)
dfs_rpeaks = pd.concat(dfs_rpeaks).to_csv("Rpeaks.csv", index=False)

# Quick test
#import neurokit2 as nk
#nk.events_plot(anno["Rpeaks"][anno["Rpeaks"] <= 1000], data["ECG"][0:1002])

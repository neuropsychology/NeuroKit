# -*- coding: utf-8 -*-
"""Script for downloading, formatting and saving the GUDB database (https://github.com/berndporr/ECG-GUDB).

It contains ECGs from 25 subjects. Each subject was recorded performing 5 different tasks for two minutes:
- sitting
- a maths test on a tablet
- walking on a treadmill
- running on a treadmill
- using a hand bike

The sampling rate is 250Hz for all experiments.

Credits and citation:
- Howell, L., & Porr, B. (2018). High precision ECG Database with annotated R peaks,
  recorded and filmed under realistic conditions.
"""
import pandas as pd
import ecg_gudb_database


dfs_ecg = []
dfs_rpeaks = []

for participant in range(25):
    print("Participant: " + str(participant+1) + "/25")
    for i, experiment in enumerate(ecg_gudb_database.GUDb.experiments):
        print("  - Condition " + str(i+1) + "/5")
        # creating class which loads the experiment
        ecg_class = ecg_gudb_database.GUDb(participant, experiment)

        # Chest Strap Data - only download if R-peaks annotations are available
        if ecg_class.anno_cs_exists:

            data = pd.DataFrame({"ECG": ecg_class.cs_V2_V1})
            data["Participant"] = "GUDB_%.2i" %(participant)
            data["Sample"] = range(len(data))
            data["Sampling_Rate"] = 250
            data["Database"] = "GUDB_" + experiment

            # getting annotations
            anno = pd.DataFrame({"Rpeaks": ecg_class.anno_cs})
            anno["Participant"] = "GUDB_%.2i" %(participant)
            anno["Sampling_Rate"] = 250
            anno["Database"] = "GUDB_" + experiment

            # Store with the rest
            dfs_ecg.append(data)
            dfs_rpeaks.append(anno)

        # Einthoven leads
#        if ecg_class.anno_cables_exists:
#            cables_anno = ecg_class.anno_cables
#            einthoven_i = ecg_class.einthoven_I
#            einthoven_ii = ecg_class.einthoven_II
#            einthoven_iii = ecg_class.einthoven_III



# Save
df_ecg = pd.concat(dfs_ecg).to_csv("ECGs.csv", index=False)
dfs_rpeaks = pd.concat(dfs_rpeaks).to_csv("Rpeaks.csv", index=False)

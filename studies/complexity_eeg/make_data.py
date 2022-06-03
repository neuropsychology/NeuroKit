import os

import mne
import numpy as np
import pandas as pd

import neurokit2 as nk

# =============================================================================
# Functions
# =============================================================================
# participant = "EEG_Cat_Study4_Resting_S1.bdf"
# path = "../../data/rs_eeg_texas/data/"
# channel = "Fp1"
# method = "rosenstein1993"


def optimize_delay(raw, channel="Fp1"):
    signal = raw.get_data(picks=channel)[0]
    vals = np.round(np.unique(nk.expspace(1, 40, 20)) / (raw.info["sfreq"] / 1000)).astype(int)
    vals = vals[vals > 0]

    rez_delay = pd.DataFrame()
    for method in ["fraser1986", "rosenstein1993", "rosenstein1994"]:
        delay, out = nk.complexity_delay(signal, delay_max=vals, method=method)

        rez = pd.DataFrame({"Value": out["Values"], "Score": out["Scores"]})
        rez["Method"] = out["Method"]
        rez["Metric"] = out["Metric"]
        # rez_delay["Algorithm"] = out["Algorithm"]
        rez["Channel"] = channel
        rez["Optimal"] = delay
        rez["What"] = "Delay"
        rez_delay = pd.concat([rez_delay, rez], axis=0)
    return rez_delay


# =============================================================================
# Run
# =============================================================================
datasets = [
    "../../data/lemon/lemon/",  # Path to local preprocessed LEMON dataset
    "../../data/rs_eeg_texas/data/",  # Path to local TEXAS dataset
    "../../data/srm_restingstate_eeg/eeg/",  # Path to local SRM dataset
]

rez_delay = pd.DataFrame()

for path in datasets:
    participants = os.listdir(path)

    for i, participant in enumerate(participants):
        if i > 3:
            continue
        print(f"Participant n°{i}")
        if "texas" in path:
            dataset = "Texas"
            sub = participant.split("_")[4].replace(".bdf", "")
            cond = "Alternating"
            raw = mne.io.read_raw_bdf(
                path + participant,
                eog=["LVEOG", "RVEOG", "LHEOG", "RHEOG"],
                misc=["NAS", "NFpz"],
                exclude=["M1", "M2"],
                preload=True,
                verbose=False,
            )
            raw = raw.set_montage("standard_1020")
            raw = raw.set_eeg_reference("average")

        elif "srm" in path:
            dataset = "SRM"
            cond = "Eyes-Closed"
            raw = mne.io.read_raw_fif(path + participant, verbose=False, preload=True)

        else:
            dataset = "Lemon"
            sub = participant.split("_")[0]
            if participant.split("_")[1] == "EO":
                cond = "Eyes-Open"
            else:
                cond = "Eyes-Closed"
            raw = mne.io.read_raw_fif(path + participant, verbose=False, preload=True)
        # mne.preprocessing.compute_current_source_density(raw)
        raw = raw.filter(1, 50, fir_design="firwin", verbose=False)

        args = [{"raw": raw, "channel": ch} for ch in raw.pick_types(eeg=True).ch_names]
        out = nk.parallel_run(optimize_delay, args, n_jobs=8, verbose=10)
        out = pd.concat(out)
        out["Participant"] = sub
        out["Condition"] = cond
        out["SamplingRate"] = raw.info["sfreq"]
        out["MaxFreq"] = raw.info["lowpass"]
        out["Duration"] = len(raw) / raw.info["sfreq"] / 60
        out["Dataset"] = dataset

        rez_delay = pd.concat([rez_delay, out], axis=0)
        rez_delay.to_csv("data_delay.csv", index=False)

print("===================")
print("FINISHED.")

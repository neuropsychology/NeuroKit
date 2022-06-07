import os

import mne
import numpy as np
import pandas as pd

import neurokit2 as nk

# =============================================================================
# Parameters
# =============================================================================
datasets = [
    "../../data/lemon/lemon/",  # Path to local preprocessed LEMON dataset
    # "../../data/rs_eeg_texas/data/",  # Path to local TEXAS dataset
    "../../data/srm_restingstate_eeg/eeg/",  # Path to local SRM dataset
    "../../data/testretest_restingstate_eeg/eeg/",  # Path to local testrestest dataset
    "C:/Dropbox/RECHERCHE/Studies/RestingStateComplexity/data/eeg_sg/",
    "C:/Dropbox/RECHERCHE/Studies/RestingStateComplexity/data/eeg_fr/",
]


# =============================================================================
# Functions
# =============================================================================
# participant = "EEG_Cat_Study4_Resting_S1.bdf"
# path = "../../data/rs_eeg_texas/data/"
# channel = "Fp1"
# method = "rosenstein1993"


def optimize_delay(raw, channel="Fp1"):
    signal = raw.get_data(picks=channel)[0]
    vals = np.unique(np.round(nk.expspace(1, 80, 60) / (1000 / raw.info["sfreq"]), 0)).astype(int)
    vals = vals[vals > 0]
    rez_delay = pd.DataFrame()
    for method in ["fraser1986", "mi2"]:
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


def optimize_dimension(raw, channel="Fp1"):
    signal = raw.get_data(picks=channel)[0]
    dim, out = nk.complexity_dimension(
        signal,
        dimension_max=5,
        delay=np.round(30 / (1000 / raw.info["sfreq"]), 0).astype(int),
        method="afn",
    )

    rez = pd.DataFrame({"Value": out["Values"], "Score": out["E1"]})
    rez["Method"] = "AFN"
    rez["Channel"] = channel
    rez["Optimal"] = dim
    rez["What"] = "Dimension"
    return rez


def read_raw(path, participant):
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

    elif "srm" in path:
        dataset = "SRM"
        cond = "Eyes-Closed"
        sub = participant.split("_")[0]
        raw = mne.io.read_raw_fif(path + participant, verbose=False, preload=True)
    elif "testretest" in path:
        dataset = "Wang (2022)"
        sub = participant.split("_")[0]
        if "eyesopen" in participant:
            cond = "Eyes-Open"
        else:
            cond = "Eyes-Closed"
        raw = mne.io.read_raw_fif(path + participant, verbose=False, preload=True)
    elif "RestingStateComplexity" in path:
        pass
        # os.listdir(path)
        # dataset = "SRM"
        # cond = "Eyes-Closed"
        # sub = participant.split("_")[0]
        # raw = mne.io.read_raw_fif(path + participant, verbose=False, preload=True)
    else:
        dataset = "Lemon"
        sub = participant.split("_")[0]
        if participant.split("_")[1] == "EO":
            cond = "Eyes-Open"
        else:
            cond = "Eyes-Closed"
        raw = mne.io.read_raw_fif(path + participant, verbose=False, preload=True)
    raw = raw.set_eeg_reference("average")
    # raw = mne.preprocessing.compute_current_source_density(raw)
    orig_filtering = f"{raw.info['highpass']}-{raw.info['lowpass']}"
    raw = raw.filter(1, 50, fir_design="firwin", verbose=False)
    return raw, sub, cond, orig_filtering, dataset


# =============================================================================
# Delay
# =============================================================================
rez_delay = pd.DataFrame()
# rez_delay = pd.read_csv("data_delay.csv")

for path in datasets:
    participants = os.listdir(path)

    for i, participant in enumerate(participants):
        if i < 0 or i > 3:
            continue
        print(f"Participant n°{i} (path: {path})")

        raw, sub, cond, orig_filtering, dataset = read_raw(path, participant)
        args = [{"raw": raw, "channel": ch} for ch in raw.pick_types(eeg=True).ch_names]
        out = nk.parallel_run(optimize_delay, args, n_jobs=8, verbose=10)
        out = pd.concat(out)

        out["Participant"] = sub
        out["Condition"] = cond
        out["Sampling_Rate"] = raw.info["sfreq"]
        out["Lowpass"] = raw.info["lowpass"]
        out["Original_Frequencies"] = orig_filtering
        out["Duration"] = len(raw) / raw.info["sfreq"] / 60
        out["Dataset"] = dataset

        rez_delay = pd.concat([rez_delay, out], axis=0)
        rez_delay.to_csv("data_delay.csv", index=False)

print("===================")
print("FINISHED.")

# =============================================================================
# Attractors
# =============================================================================
attractors = pd.DataFrame()
for path in datasets:
    participants = os.listdir(path)[2]
    raw, _, _, _, dataset = read_raw(path, participants)
    raw = raw.crop(tmin=30, tmax=90)
    for channel in ["Fz", "Pz"]:
        signal = nk.standardize(raw.get_data(picks=channel)[0])
        # nk.signal_psd(signal, sampling_rate=raw.info["sfreq"], max_frequency=100, show=True)
        d = np.round(np.array([30]) / (1000 / raw.info["sfreq"]), 0).astype(int)
        for i, delay in enumerate(d):
            if delay == 0:
                continue
            data = nk.complexity_embedding(signal, delay=delay, dimension=4)
            data = pd.DataFrame(data, columns=["x", "y", "z", "c"])
            data["Dataset"] = dataset
            data["Sampling_Rate"] = raw.info["sfreq"]
            data["Delay"] = delay
            data["Delay_Type"] = i
            data["Channel"] = channel
            data["Time"] = raw.times[0 : len(data)]
            attractors = pd.concat([attractors, data], axis=0)
attractors.to_csv("data_attractor.csv", index=False)

attractors = pd.DataFrame()
for path in datasets:
    participants = os.listdir(path)[2]
    raw, _, _, _, dataset = read_raw(path, participants)
    raw = raw.crop(tmin=85, tmax=90)
    signal = nk.standardize(raw.get_data(picks="Fz")[0])
    for freq in range(1, 51):
        delay = np.round(freq / (1000 / raw.info["sfreq"]), 0).astype(int)
        if delay == 0:
            continue
        data = nk.complexity_embedding(signal, delay=delay, dimension=2)
        data = pd.DataFrame(data, columns=["x", "y"])
        data["Dataset"] = dataset
        data["Delay"] = delay
        data["Period"] = freq
        data["Time"] = raw.times[0 : len(data)]
        attractors = pd.concat([attractors, data], axis=0)
attractors.to_csv("data_attractor_anim.csv", index=False)


# =============================================================================
# Dimension
# =============================================================================
rez_dim = pd.DataFrame()

for path in datasets:
    participants = os.listdir(path)

    for i, participant in enumerate(participants):
        if i < 0 or i > 1:
            continue
        print(f"Participant n°{i} (path: {path})")

        raw, sub, cond, orig_filtering, dataset = read_raw(path, participant)
        args = [{"raw": raw, "channel": ch} for ch in raw.pick_types(eeg=True).ch_names]
        out = nk.parallel_run(optimize_dimension, args, n_jobs=8, verbose=10)
        out = pd.concat(out)

        out["Participant"] = sub
        out["Condition"] = cond
        out["Sampling_Rate"] = raw.info["sfreq"]
        out["Lowpass"] = raw.info["lowpass"]
        out["Original_Frequencies"] = orig_filtering
        out["Duration"] = len(raw) / raw.info["sfreq"] / 60
        out["Dataset"] = dataset

        rez_dim = pd.concat([rez_dim, out], axis=0)
        rez_dim.to_csv("data_dimension.csv", index=False)

print("===================")
print("FINISHED.")

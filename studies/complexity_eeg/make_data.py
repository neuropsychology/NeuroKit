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
    "../../data/rebel_eeg_restingstate_sg/eeg/",
    "../../data/rebel_eeg_restingstate_fr/eeg/",
]


# =============================================================================
# Functions
# =============================================================================
# channel = "Fp1"


def optimize_delay(raw, channel="Fp1"):
    signal = raw.get_data(picks=channel)[0]
    vals = np.unique(np.round(np.linspace(1, 80, 80) / (1000 / raw.info["sfreq"]), 0)).astype(int)
    vals = vals[vals > 0]
    delay, out = nk.complexity_delay(signal, delay_max=vals, method="fraser1986")

    rez = pd.DataFrame({"Value": out["Values"], "Score": out["Scores"]})
    rez["Method"] = out["Method"]
    rez["Metric"] = out["Metric"]
    # rez_delay["Algorithm"] = out["Algorithm"]
    rez["Channel"] = channel
    rez["Optimal"] = delay
    rez["What"] = "Delay"
    return rez


def optimize_dimension(raw, channel="Fp1"):
    signal = raw.get_data(picks=channel)[0]
    dim, out = nk.complexity_dimension(
        signal,
        dimension_max=5,
        delay=np.round(27 / (1000 / raw.info["sfreq"]), 0).astype(int),
        method="afn",
    )

    rez = pd.DataFrame({"Value": out["Values"], "Score": out["E1"]})
    rez["Method"] = "AFN"
    rez["Channel"] = channel
    rez["Optimal"] = dim
    rez["What"] = "Dimension"
    return rez


def optimize_tolerance(raw, channel="Fp1"):
    signal = raw.get_data(picks=channel)[0]
    signal = nk.standardize(signal)
    out = pd.DataFrame()
    for method in ["maxApEn", "recurrence"]:

        r, info = nk.complexity_tolerance(
            signal,
            delay=np.round(27 / (1000 / raw.info["sfreq"]), 0).astype(int),
            dimension=5,
            method=method,
            r_range=np.linspace(0.002, 2, 10),
            show=True,
        )

        rez = pd.DataFrame({"Value": info["Values"], "Score": info["Scores"]})
        rez["Method"] = method
        rez["Channel"] = channel
        rez["Optimal"] = r
        rez["What"] = "Dimension"
        out = pd.concat([out, rez], axis=0)

    return out


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
    elif "rebel_eeg_restingstate" in path:
        if "_sg" in path:
            dataset = "Resting-State (SG)"
        else:
            dataset = "Resting-State (FR)"
        cond = "Eyes-Closed"
        sub = participant
        raw = mne.io.read_raw_fif(path + participant, verbose=False, preload=True)
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


def compute_complexity(raw, channel="Fp1"):
    signal = raw.get_data(picks=channel)[0]
    signal = nk.standardize(signal)

    delay = np.round(27 / (1000 / raw.info["sfreq"]), 0).astype(int)
    m = 5
    r, _ = nk.complexity_tolerance(signal, dimension=m, method="NeuroKit")

    rez = pd.DataFrame({"Channel": [channel]})
    rez["SFD"], _ = nk.fractal_sevcik(signal)  # Change ShanEn D by SFD
    rez["MSWPEn"], _ = nk.entropy_multiscale(
        signal, scale="default", dimension=m, tolerance=r, method="MSWPEn"
    )
    rez["CWPEn"], _ = nk.entropy_permutation(
        signal,
        delay=delay,
        dimension=m,
        tolerance=r,
        weighted=True,
        conditional=True,
    )
    rez["AttEn"], _ = nk.entropy_attention(
        signal,
    )
    rez["SVDEn"], _ = nk.entropy_svd(signal, delay=delay, dimension=m)
    rez["Hjorth"], _ = nk.complexity_hjorth(signal)
    rez["FDNLD"], _ = nk.fractal_nld(signal)

    mfdfa, _ = nk.fractal_dfa(signal, multifractal=True)
    rez["MFDFA_Width"] = mfdfa["Width"]
    rez["MFDFA_Max"] = mfdfa["Max"]
    rez["MFDFA_Mean"] = mfdfa["Mean"]
    rez["MFDFA_Increment"] = mfdfa["Increment"]

    return rez


# =============================================================================
# Delay
# =============================================================================
# rez_delay = pd.DataFrame()

# for path in datasets:
#     participants = os.listdir(path)

#     for i, participant in enumerate(participants):
#         if i < 0 or i > 3:
#             continue
#         print(f"Participant n°{i} (path: {path})")

#         raw, sub, cond, orig_filtering, dataset = read_raw(path, participant)
#         args = [{"raw": raw, "channel": ch} for ch in raw.pick_types(eeg=True).ch_names]
#         for i in args:
#             optimize_delay(i["raw"], i["channel"])
#         out = nk.parallel_run(optimize_delay, args, n_jobs=1, verbose=10)
#         out = pd.concat(out)

#         out["Participant"] = sub
#         out["Condition"] = cond
#         out["Sampling_Rate"] = raw.info["sfreq"]
#         out["Lowpass"] = raw.info["lowpass"]
#         out["Original_Frequencies"] = orig_filtering
#         out["Duration"] = len(raw) / raw.info["sfreq"] / 60
#         out["Dataset"] = dataset

#         rez_delay = pd.concat([rez_delay, out], axis=0)
#         rez_delay.to_csv("data_delay.csv", index=False)

# print("===================")
# print("FINISHED.")

# =============================================================================
# Attractors
# =============================================================================
attractors = pd.DataFrame()
for path in datasets:
    participants = os.listdir(path)[2]
    raw, _, _, _, dataset = read_raw(path, participants)
    raw = raw.crop(tmin=30, tmax=90)
    for channel in ["Cz", "Fz", "AFz", "Pz", "Oz"]:  # raw.ch_names
        if channel not in raw.ch_names:
            print(f"Channel: {channel}, Dataset: {dataset}")
            continue
        signal = nk.standardize(raw.get_data(picks=channel)[0])
        # if "testretest" in path:
        #     nk.signal_psd(signal, sampling_rate=raw.info["sfreq"], max_frequency=100, show=True)
        d = np.round(np.array([27]) / (1000 / raw.info["sfreq"]), 0).astype(int)
        for i, delay in enumerate(d):
            if delay == 0:
                continue
            data = nk.complexity_embedding(signal, delay=delay, dimension=4)
            data = pd.DataFrame(data, columns=["x", "y", "z", "c"])
            data["Dataset"] = dataset
            data["Sampling_Rate"] = raw.info["sfreq"]
            data["Delay"] = delay
            data["Delay_Type"] = i
            data["Channel"] = "Fz" if channel == "AFz" else channel
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


# =============================================================================
# Tolerance
# =============================================================================
rez_r = pd.DataFrame()

for path in datasets:
    participants = os.listdir(path)

    for i, participant in enumerate(participants):
        if i < 0 or i > 1:
            continue
        print(f"Participant n°{i} (path: {path})")

        raw, sub, cond, orig_filtering, dataset = read_raw(path, participant)
        args = [{"raw": raw, "channel": ch} for ch in raw.pick_types(eeg=True).ch_names]
        out = nk.parallel_run(optimize_tolerance, args, n_jobs=8, verbose=10)
        out = pd.concat(out)

        out["Participant"] = sub
        out["Condition"] = cond
        out["Sampling_Rate"] = raw.info["sfreq"]
        out["Lowpass"] = raw.info["lowpass"]
        out["Original_Frequencies"] = orig_filtering
        out["Duration"] = len(raw) / raw.info["sfreq"] / 60
        out["Dataset"] = dataset

        rez_r = pd.concat([rez_r, out], axis=0)
        rez_r.to_csv("data_tolerance.csv", index=False)

print("===================")
print("FINISHED.")


# =============================================================================
# Clustering
# =============================================================================
rez_complexity = pd.DataFrame()

for path in datasets:
    participants = os.listdir(path)

    for i, participant in enumerate(participants):
        if i < 0 or i > 100:
            continue
        print(f"Participant n°{i} (path: {path})")

        raw, sub, cond, orig_filtering, dataset = read_raw(path, participant)
        args = [{"raw": raw, "channel": ch} for ch in raw.pick_types(eeg=True).ch_names]
        out = nk.parallel_run(compute_complexity, args, n_jobs=1, verbose=10)
        out = pd.concat(out)

        out["Participant"] = sub
        out["Condition"] = cond
        out["Dataset"] = dataset

        rez_complexity = pd.concat([rez_complexity, out], axis=0)
        rez_complexity.to_csv("data_complexity.csv", index=False)
print("===================")
print("FINISHED.")

import pandas as pd
import neurokit2 as nk


# Load ECGs
ecgs_gudb = pd.read_csv("../../data/gudb/ECGs.csv")
ecgs_mit1 = pd.read_csv("../../data/mit_arrhythmia/ECGs.csv")
ecgs_mit2 = pd.read_csv("../../data/mit_normal/ECGs.csv")

# Load True R-peaks location
rpeaks_gudb = pd.read_csv("../../data/gudb/Rpeaks.csv")
rpeaks_mit1 = pd.read_csv("../../data/mit_arrhythmia/Rpeaks.csv")
rpeaks_mit2 = pd.read_csv("../../data/mit_normal/Rpeaks.csv")

# Concatenate
#ecgs = pd.concat([ecgs_gudb, ecgs_mit1, ecgs_mit2], ignore_index=True)
#rpeaks = pd.concat([rpeaks_gudb, rpeaks_mit1, rpeaks_mit2], ignore_index=True)
ecgs = [ecgs_gudb, ecgs_mit1, ecgs_mit2]
rpeaks = [rpeaks_gudb, rpeaks_mit1, rpeaks_mit2]










# =============================================================================
# Study 1
# =============================================================================


#def neurokit(ecg, sampling_rate):
#    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
#    return info["ECG_R_Peaks"]
#
#def pantompkins1985(ecg, sampling_rate):
#    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="pantompkins1985")
#    return info["ECG_R_Peaks"]
#
#def hamilton2002(ecg, sampling_rate):
#    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="hamilton2002")
#    return info["ECG_R_Peaks"]
#
#def martinez2003(ecg, sampling_rate):
#    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="martinez2003")
#    return info["ECG_R_Peaks"]
#
#def christov2004(ecg, sampling_rate):
#    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="christov2004")
#    return info["ECG_R_Peaks"]
#
#def gamboa2008(ecg, sampling_rate):
#    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="gamboa2008")
#    return info["ECG_R_Peaks"]
#
#def elgendi2010(ecg, sampling_rate):
#    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="elgendi2010")
#    return info["ECG_R_Peaks"]
#
#def engzeemod2012(ecg, sampling_rate):
#    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="engzeemod2012")
#    return info["ECG_R_Peaks"]
#
#def kalidas2017(ecg, sampling_rate):
#    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="kalidas2017")
#    return info["ECG_R_Peaks"]
#
#def rodrigues2020(ecg, sampling_rate):
#    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="rodrigues2020")
#    return info["ECG_R_Peaks"]
#
#
#
#
#
#
#results = []
#for method in [neurokit, pantompkins1985, hamilton2002, martinez2003, christov2004,
#               gamboa2008, elgendi2010, engzeemod2012, kalidas2017, rodrigues2020]:
#    print(method.__name__)
#    for i in range(len(ecgs)):
#        print("  - " + str(i))
#        result = nk.benchmark_ecg_preprocessing(method, ecgs[i], rpeaks[i])
#        result["Method"] = method.__name__
#        results.append(result)
#results = pd.concat(results).reset_index(drop=True)
#
#results.to_csv("data_detectors.csv", index=False)












# =============================================================================
# Study 2
# =============================================================================
#def none(ecg, sampling_rate):
#    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
#    return info["ECG_R_Peaks"]
#
#def mean_removal(ecg, sampling_rate):
#    ecg = nk.signal_detrend(ecg, order=0)
#    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
#    return info["ECG_R_Peaks"]
#
#def standardization(ecg, sampling_rate):
#    ecg = nk.standardize(ecg)
#    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
#    return info["ECG_R_Peaks"]
#
#results = []
#for method in [none, mean_removal, standardization]:
#    print(method.__name__)
#    for i in range(len(ecgs)):
#        print("  - " + str(i))
#        result = nk.benchmark_ecg_preprocessing(method, ecgs[i], rpeaks[i])
#        result["Method"] = method.__name__
#        results.append(result)
#results = pd.concat(results).reset_index(drop=True)
#
#results.to_csv("data_normalization.csv", index=False)





# =============================================================================
# Study 3
# =============================================================================
def none(ecg, sampling_rate):
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
    return info["ECG_R_Peaks"]

# Detrending-based
def polylength(ecg, sampling_rate):
    length = len(ecg) / sampling_rate
    ecg = nk.signal_detrend(ecg, method="polynomial", order=int(length / 2))
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
    return info["ECG_R_Peaks"]

def tarvainen(ecg, sampling_rate):
    ecg = nk.signal_detrend(ecg, method="tarvainen2002")
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
    return info["ECG_R_Peaks"]

def locreg(ecg, sampling_rate):
    ecg = nk.signal_detrend(ecg,
                            method="locreg",
                            window=0.5*sampling_rate,
                            stepsize=0.02*sampling_rate)
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
    return info["ECG_R_Peaks"]

def rollingz(ecg, sampling_rate):
    ecg = nk.standardize(ecg, window=sampling_rate*2)
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
    return info["ECG_R_Peaks"]


results = []
for method in [none, polylength, tarvainen, locreg, rollingz]:
    print(method.__name__)
    for i in range(len(ecgs)):
        print("  - " + str(i))
        result = nk.benchmark_ecg_preprocessing(method, ecgs[i], rpeaks[i])
        result["Method"] = method.__name__
        results.append(result)
results = pd.concat(results).reset_index(drop=True)

results.to_csv("data_lowfreq.csv", index=False)
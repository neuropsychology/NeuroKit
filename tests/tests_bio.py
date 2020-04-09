import neurokit2 as nk
import numpy as np

def test_bio_process():

    sampling_rate = 1000

    # Create data
    ecg = nk.ecg_simulate(duration=30, sampling_rate=sampling_rate)
    rsp = nk.rsp_simulate(duration=30, sampling_rate=sampling_rate)
    eda = nk.eda_simulate(duration=30, sampling_rate=sampling_rate, scr_number=3)
    emg = nk.emg_simulate(duration=30, sampling_rate=sampling_rate, burst_number=3)

    bio_df, bio_info = nk.bio_process(ecg=ecg,
                                      rsp=rsp,
                                      eda=eda,
                                      emg=emg,
                                      sampling_rate=sampling_rate)

    # SCR components
    scr = [val for key, val in bio_info.items() if "SCR" in key]
    assert all(len(elem) == len(scr[0]) for elem in scr)
    assert all(bio_info["SCR_Onsets"] < bio_info["SCR_Peaks"])
    assert all(bio_info["SCR_Peaks"] < bio_info["SCR_Recovery"])

    # RSP
    assert all(bio_info["RSP_Peaks"] > bio_info["RSP_Troughs"])
    assert len(bio_info["RSP_Peaks"]) == len(bio_info["RSP_Troughs"])

    # EMG
    assert all(bio_info["EMG_Offsets"] > bio_info["EMG_Onsets"])
    assert len(bio_info["EMG_Offsets"] == len(bio_info["EMG_Onsets"]))

    assert all(elem in ['ECG_Raw', 'ECG_Clean', 'ECG_Rate',
                        'ECG_Quality', 'ECG_R_Peaks',
                        "ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks",
                        "ECG_T_Peaks", "ECG_P_Onsets", "ECG_T_Offsets",
                        "ECG_Atrial_Phase", "ECG_Ventricular_Phase",
                        "ECG_Atrial_PhaseCompletion",
                        "ECG_Ventricular_PhaseCompletion",
                        'RSP_Raw', 'RSP_Clean', 'RSP_Amplitude', 'RSP_Rate',
                        'RSP_Phase', 'RSP_PhaseCompletion',
                        'RSP_Peaks', 'RSP_Troughs',
                        'EDA_Raw', 'EDA_Clean', 'EDA_Tonic', 'EDA_Phasic',
                        'SCR_Onsets', 'SCR_Peaks', 'SCR_Height', 'SCR_Amplitude',
                        'SCR_RiseTime', 'SCR_Recovery', 'SCR_RecoveryTime',
                        'EMG_Raw', 'EMG_Clean', 'EMG_Amplitude', 'EMG_Activity',
                        'EMG_Onsets', 'EMG_Offsets', 'RSA_P2T']
               for elem in np.array(bio_df.columns.values, dtype=str))

def test_bio_analyze():

    # Example with event-related analysis
    data = nk.data("bio_eventrelated_100hz")
    df, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"],
                              eda=data["EDA"], keep=data["Photosensor"],
                              sampling_rate=100)
    events = nk.events_find(data["Photosensor"],
                            threshold_keep='below',
                            event_conditions=["Negative",
                                              "Neutral",
                                              "Neutral",
                                              "Negative"])
    epochs = nk.epochs_create(df, events,
                              sampling_rate=100,
                              epochs_start=-0.1, epochs_end=1.9)
    event_related = nk.bio_analyze(epochs)

    assert len(event_related) == len(epochs)
    labels = [int(i) for i in event_related['Label']]
    assert labels == list(np.arange(1, len(epochs)+1))
    assert all(elem in ['ECG_Rate_Max', 'ECG_Rate_Min', 'ECG_Rate_Mean',
                        'ECG_Rate_Max_Time', 'ECG_Rate_Min_Time',
                        'ECG_Rate_Trend_Quadratic', 'ECG_Rate_Trend_Linear',
                        'ECG_Rate_Trend_R2', 'ECG_Atrial_Phase',
                        'ECG_Atrial_PhaseCompletion', 'ECG_Ventricular_Phase',
                        'ECG_Ventricular_PhaseCompletion', 'ECG_Quality_Mean',
                        'RSP_Rate_Max',  'RSP_Rate_Min',
                        'RSP_Rate_Mean', 'RSP_Rate_Max_Time', 'RSP_Rate_Min_Time',
                        'RSP_Amplitude_Max', 'RSP_Amplitude_Min', 'RSP_Amplitude_Mean',
                        'RSP_Phase', 'RSP_PhaseCompletion', 'EDA_Activation',
                        'EDA_Peak_Amplitude', 'EDA_Peak_Amplitude_Time', 'EDA_RiseTime',
                        'EDA_RecoveryTime', 'RSA_P2T', 'Label', 'Condition']
               for elem in np.array(event_related.columns.values, dtype=str))

    # Example with interval-related analysis
    data = nk.data("bio_resting_8min_100hz")
    df, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"],
                              eda=data["EDA"], sampling_rate=100)
    interval_related = nk.bio_analyze(df)

    assert len(interval_related) == 1
    assert all(elem in ['ECG_Rate_Mean', 'ECG_HRV_RMSSD', 'ECG_HRV_MeanNN',
                        'ECG_HRV_SDNN', 'ECG_HRV_SDSD', 'ECG_HRV_CVNN',
                        'ECG_HRV_CVSD', 'ECG_HRV_MedianNN',
                        'ECG_HRV_MadNN', 'ECG_HRV_MCVNN',
                        'ECG_HRV_pNN50', 'ECG_HRV_pNN20',
                        'ECG_HRV_TINN', 'ECG_HRV_HTI',
                        'ECG_HRV_ULF', 'ECG_HRV_VLF',
                        'ECG_HRV_LF', 'ECG_HRV_HF', 'ECG_HRV_VHF', 'ECG_HRV_LFHF',
                        'ECG_HRV_LFn', 'ECG_HRV_HFn', 'ECG_HRV_LnHF', 'ECG_HRV_SD1',
                        'ECG_HRV_SD2', 'ECG_HRV_SD2SD1', 'ECG_HRV_CSI', 'ECG_HRV_CVI',
                        'ECG_HRV_CSI_Modified', 'ECG_HRV_SampEn', 'RSP_Rate_Mean',
                        'RSP_Amplitude_Mean', 'RSP_RRV_SDBB', 'RSP_RRV_RMSSD',
                        'RSP_RRV_SDSD', 'RSP_RRV_VLF', 'RSP_RRV_LF', 'RSP_RRV_HF',
                        'RSP_RRV_LFHF', 'RSP_RRV_LFn', 'RSP_RRV_HFn', 'RSP_RRV_SD1',
                        'RSP_RRV_SD2', 'RSP_RRV_SD2SD1', 'RSP_RRV_ApEn',
                        'RSP_RRV_SampEn', 'RSP_RRV_DFA', 'RSA_P2T_Mean',
                        'RSA_P2T_Mean_log', 'RSA_P2T_SD',
                        'RSA_P2T_NoRSA', 'RSA_PorgesBohrer',
                        'EDA_Peaks_N', 'EDA_Peaks_Amplitude_Mean']
               for elem in np.array(interval_related.columns.values, dtype=str))

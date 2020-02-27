import neurokit2 as nk
import numpy as np

def test_bio_process():

    sampling_rate = 1000

    # Create data
    ecg = nk.ecg_simulate(duration=30, sampling_rate=sampling_rate)
    rsp = nk.rsp_simulate(duration=30, sampling_rate=sampling_rate)
    eda = nk.eda_simulate(duration=30, sampling_rate=sampling_rate, n_scr=3)
    emg = nk.emg_simulate(duration=30, sampling_rate=sampling_rate, n_bursts=3)

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

    assert all(elem in ['ECG_Raw', 'ECG_Clean', 'ECG_Rate', 'ECG_R_Peaks',
                        'RSP_Raw', 'RSP_Clean', 'RSP_Inspiration',
                        'RSP_Amplitude', 'RSP_Rate', 'RSP_Peaks', 'RSP_Troughs',
                        'EDA_Raw', 'EDA_Clean', 'EDA_Tonic', 'EDA_Phasic',
                        'SCR_Onsets', 'SCR_Peaks', 'SCR_Height', 'SCR_Amplitude',
                        'SCR_RiseTime', 'SCR_Recovery', 'SCR_RecoveryTime',
                        'EMG_Raw', 'EMG_Clean', 'EMG_Amplitude', 'EMG_Activity',
                        'EMG_Onsets', 'EMG_Offsets']
               for elem in np.array(bio_df.columns.values, dtype=str))

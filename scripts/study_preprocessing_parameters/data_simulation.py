import numpy as np
import pandas as pd
import neurokit2 as nk


def rsp_custom_process(rsp, sampling_rate=1000):

    clean = signal_detrend(rsp_signal, order=1)
    clean = signal_filter(clean, sampling_rate=sampling_rate,
                          lowcut=None, highcut=2,
                          method="butterworth", butterworth_order=5)

    extrema_signal, info = rsp_findpeaks(filtered_rsp,
                                         sampling_rate=sampling_rate,
                                         outlier_threshold=0.3)

    rate = rsp_rate(extrema_signal, sampling_rate=sampling_rate)

    signals = pd.DataFrame({"RSP_Raw": rsp_signal,
                            "RSP_Filtered": filtered_rsp})
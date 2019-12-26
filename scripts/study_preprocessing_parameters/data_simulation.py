import numpy as np
import pandas as pd
import neurokit2 as nk


#def rsp_custom_process(rsp, sampling_rate=1000):
#
#    clean = nk.signal_detrend(rsp_signal, order=1)
#    clean = nk.signal_filter(clean, sampling_rate=sampling_rate,
#                          lowcut=None, highcut=2,
#                          method="butterworth", butterworth_order=5)
#
#    extrema_signal, info = nk.rsp_findpeaks(clean,
#                                            sampling_rate=sampling_rate,
#                                            outlier_threshold=0.3)
#
#    rate = nk.rsp_rate(extrema_signal, sampling_rate=sampling_rate)

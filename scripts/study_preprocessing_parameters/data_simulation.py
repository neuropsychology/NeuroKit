import numpy as np
import pandas as pd
import neurokit2 as nk

import matplotlib.pyplot as plt

# =============================================================================
# RSP - Functions
# =============================================================================
def rsp_generate(duration=90, sampling_rate=1000, respiratory_rate=15, method="simple"):

    if method == "Simple":
        actual_method = "sinusoidal"
    else:
        actual_method = "breathmetrics"

    rsp = nk.rsp_simulate(duration=duration, sampling_rate=sampling_rate, respiratory_rate=respiratory_rate, noise=0, method=actual_method)

    info = {"Duration": [duration],
            "Sampling_Rate": [sampling_rate],
            "Respiratory_Rate": [respiratory_rate],
            "Simulation": [method]}

    return rsp, info



def rsp_distord(rsp, info, noise_amplitude=0.1, noise_frequency=100):
    distorted = nk.signal_distord(rsp,
                                  noise_amplitude=noise_amplitude,
                                  noise_frequency=noise_frequency,
                                  noise_shape="laplace")
    info["Noise_Amplitude"] = [noise_amplitude]
    info["Noise_Frequency"] = [noise_frequency]
    return distorted, info



def rsp_custom_process(rsp, info, detrend_first=True, detrend_order=0):
    sampling_rate = info["Sampling_Rate"][0]

    if detrend_first is True:
        rsp = nk.signal_detrend(rsp, order=detrend_order)

    rsp = nk.signal_filter(rsp,
                             sampling_rate=sampling_rate,
                             lowcut=None,
                             highcut=2,
                             method="butterworth",
                             butterworth_order=5)

    if detrend_first is False:
        rsp = nk.signal_detrend(rsp, order=detrend_order)

    extrema_signal, _ = nk.rsp_findpeaks(rsp, outlier_threshold=0.3)

    rate = nk.rsp_rate(extrema_signal, sampling_rate=sampling_rate)["RSP_Rate"]

    info["Detrend_Order"] = [detrend_order]
    info["Detrend_First"] = [detrend_first]
    return rate, info







def rsp_quality(rate, info, noise_amplitude=0.1, noise_frequency=100):
    diff = info["Respiratory_Rate"][0] - rate
    info["Difference_Mean"] = np.mean(diff)
    info["Difference_SD"] = np.std(diff, ddof=1)

    data = pd.DataFrame.from_dict(info)
    return data


# =============================================================================
# RSP - Run
# =============================================================================
all_data = []
for simulation in ["Simple", "Complex"]:
    for noise_amplitude in np.linspace(0.01, 1, 40):
        for noise_frequency in np.linspace(1, 100, 40):
            for detrend_first in [True, False]:
                for detrend_order in [0, 1, 2, 3, 4]:
                    rsp, info = rsp_generate(duration=90, sampling_rate=1000, respiratory_rate=15, method=simulation)
                    distorted, info = rsp_distord(rsp, info, noise_amplitude=noise_amplitude, noise_frequency=noise_frequency)
                    rate, info = rsp_custom_process(distorted, info, detrend_first=detrend_first, detrend_order=detrend_order)
                    data = rsp_quality(rate, info, noise_amplitude=0.1, noise_frequency=100)
                    all_data += [data]
data = pd.concat(all_data)
data.to_csv("data.csv")

# Check
fig, axes = plt.subplots(nrows=2, ncols=2)

data.plot.scatter(x="Noise_Amplitude", y="Difference_Mean", color='r', ax=axes[0,0])
data.plot.scatter(x="Noise_Amplitude", y="Difference_SD", color='r', ax=axes[1,0])
data.plot.scatter(x="Noise_Frequency", y="Difference_Mean", color='g', ax=axes[0,1])
data.plot.scatter(x="Noise_Frequency", y="Difference_SD", color='g', ax=axes[1,1])

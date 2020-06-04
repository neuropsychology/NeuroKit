import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import neurokit2 as nk


# =============================================================================
# RSP - Functions
# =============================================================================
def rsp_generate(duration=90, sampling_rate=1000, respiratory_rate=None, method="Complex"):

    if respiratory_rate is None:
        respiratory_rate = np.random.randint(10, 25)

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



def rsp_distord(rsp, info, noise_amplitude=0.1, noise_frequency=None):

    # Frequency
    if noise_frequency is None:
        n_frequencies = np.random.randint(1, 5)
        noise_frequency = np.random.uniform(0.05, 100, n_frequencies) # To max frequency
        noise_frequency = np.round(noise_frequency, 2).tolist()

    distorted = nk.signal_distord(rsp,
                                  noise_amplitude=noise_amplitude,
                                  noise_frequency=noise_frequency,
                                  noise_shape="laplace",
                                  powerline_frequency=50,
                                  powerline_amplitude=0.1)

    # Artifacts
    artifacts_frequency = np.random.uniform(0.05, 1)
    artifacts_amplitude = np.random.uniform(0, 2)
    distorted = nk.signal_distord(distorted,
                                  noise_amplitude=0,
                                  powerline_amplitude=0,
                                  artifacts_amplitude=artifacts_amplitude,
                                  artifacts_frequency=artifacts_frequency)

    info["Noise_Amplitude"] = [noise_amplitude]
    info["Noise_Frequency"] = [noise_frequency]
    info["Noise_Number"] = [n_frequencies]
    info["Artifacts_Amplitude"] = [artifacts_amplitude]
    info["Artifacts_Frequency"] = [artifacts_frequency]

    return distorted, info



def rsp_custom_process(distorted, info, detrend_position="First", detrend_method="polynomial", detrend_order=0, detrend_regularization=500, detrend_alpha=0.75, filter_type="None", filter_order=5, filter_lowcut=None, filter_highcut=None):
    sampling_rate = info["Sampling_Rate"][0]

    if detrend_position in ["First", 'Both']:
        distorted = nk.signal_detrend(distorted,
                                      method=detrend_method,
                                      order=detrend_order,
                                      regularization=detrend_regularization,
                                      alpha=detrend_alpha)

    if filter_type != "None":
        distorted = nk.signal_filter(signal=distorted,
                                     sampling_rate=sampling_rate,
                                     lowcut=filter_lowcut,
                                     highcut=filter_highcut,
                                     method=filter_type,
                                     order=filter_order)

    if detrend_position in ["Second", 'Both']:
        distorted = nk.signal_detrend(distorted,
                                      method=detrend_method,
                                      order=int(detrend_order),
                                      regularization=detrend_regularization,
                                      alpha=detrend_alpha)
    cleaned = distorted
    extrema_signal, _ = nk.rsp_findpeaks(distorted, outlier_threshold=0)

    try:
        rate = nk.rsp_rate(peaks=extrema_signal, sampling_rate=sampling_rate)
    except ValueError:
        rate = np.full(len(distorted), np.nan)

    info["Detrend_Method"] = [detrend_method]
    info["Detrend_Order"] = [detrend_order]
    info["Detrend_Regularization"] = [detrend_regularization]
    info["Detrend_Alpha"] = [detrend_alpha]
    info["Detrend_Position"] = [detrend_position]

    info["Filter_Method"] = [filter_type]
    if filter_type in ["Butterworth", "Bessel"]:
        info["Filter_Type"] = [filter_type + "_" + str(filter_order)]
    else:
        info["Filter_Type"] = [filter_type]
    info["Filter_Order"] = [filter_order]
    info["Filter_Low"] = [filter_lowcut]
    info["Filter_High"] = [filter_highcut]
    if filter_lowcut is None and filter_highcut is None:
        info["Filter_Band"] = "None"
    else:
        info["Filter_Band"] = [str(np.round(filter_lowcut, 3)) + ", " + str(np.round(filter_highcut, 3))]
    return rate, info, cleaned





def rsp_quality(rate, info, cleaned, rsp):
    diff = info["Respiratory_Rate"][0] - rate
    info["Difference_Rate_Mean"] = np.mean(diff)
    info["Difference_Rate_SD"] = np.std(diff, ddof=1)

    diff = rsp - cleaned
    info["Difference_Cleaned_Mean"] = np.mean(diff)
    info["Difference_Cleaned_SD"] = np.std(diff, ddof=1)

    data = pd.DataFrame.from_dict(info)
    return data








# =============================================================================
# RSP - Visualize noise
# =============================================================================
#rsp, info = rsp_generate(duration=60, sampling_rate=200, method="Simple")
#data = pd.DataFrame({"Original": rsp})
#for noise_amplitude in np.linspace(0.01, 1, 7):
#
#    # Noise frequency
#    distorted, _ = rsp_distord(rsp.copy(), info, noise_amplitude=noise_amplitude, noise_frequency=None)
#    data["Distorted_" + str(np.round(noise_amplitude, 2))] = distorted
#
## Visualize
#df = data[0:2000]
#for i, column in enumerate(df.drop('Original', axis=1)):
#   plt.plot(df[column], marker='', color='grey', linewidth=1, alpha=(i/len(df.columns))*0.5)
#plt.plot(df['Original'], marker='', color='red', linewidth=3)



# =============================================================================
# RSP - Filter# =============================================================================
all_data = []
for noise_amplitude in np.linspace(0.01, 1, 100):
    print("---")
    print("%.2f" %(noise_amplitude*100))
    print("---")
    respiratory_rate = np.random.uniform(10, 20)
    rsp, info = rsp_generate(duration=60, sampling_rate=200, respiratory_rate=respiratory_rate, method="Simple")
    distorted, info = rsp_distord(rsp, info, noise_amplitude=noise_amplitude, noise_frequency=None)
    print("Noise freq:", info["Noise_Frequency"][0])

    rate, info, cleaned = rsp_custom_process(distorted, info, detrend_position="None", filter_type="None")
    data = rsp_quality(rate, info, cleaned, rsp)
    all_data += [data]

    for filter_highcut in np.linspace(18/60, 80/60, 4):
        for filter_lowcut in np.linspace(4/60, 10/60, 4):
            for filter_type in ["Butterworth", "Bessel", "FIR"]:
                if filter_type == "FIR":
                    rate, info, cleaned = rsp_custom_process(distorted, info, detrend_position="None", filter_type=filter_type, filter_highcut=filter_highcut, filter_lowcut=filter_lowcut)
                    data = rsp_quality(rate, info, cleaned, rsp)
                    all_data += [data]
                else:
                    for filter_order in range(9):
                        rate, info, cleaned = rsp_custom_process(distorted, info, detrend_position="None", filter_type=filter_type, filter_order=filter_order+1, filter_highcut=filter_highcut, filter_lowcut=filter_lowcut)
                        data = rsp_quality(rate, info, cleaned, rsp)
                        all_data += [data]

    data = pd.concat(all_data)
    data.to_csv("data_RSP_filtering.csv")

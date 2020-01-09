import numpy as np
import pandas as pd
#import neurokit2 as nk

import matplotlib.pyplot as plt

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

    if noise_frequency is None:
        n_frequencies = np.random.randint(0, 10)
        noise_frequency = np.random.uniform(0.1, 200, n_frequencies) # To max frequency
        noise_frequency = np.array(noise_frequency, dtype=np.int)

    distorted = nk.signal_distord(rsp,
                                  noise_amplitude=noise_amplitude,
                                  noise_frequency=noise_frequency,
                                  noise_shape="laplace",
                                  powerline_frequency=50,
                                  powerline_amplitude=noise_amplitude)

    info["Noise_Amplitude"] = [noise_amplitude]
    info["Noise_Frequency"] = [noise_frequency]
    info["Noise_Number"] = [n_frequencies]
    return distorted, info



def rsp_custom_process(distorted, info, detrend_position="First", detrend_method="polynomial", detrend_order=0, detrend_regularization=500, detrend_alpha=0.75, filter_type="None", filter_order=5, filter_lowcut=None, filter_highcut=None):
    sampling_rate = info["Sampling_Rate"][0]

    if detrend_position in ["First", 'Both']:
        distorted = nk.signal_detrend(distorted,
                                      method=detrend_method,
                                      order=detrend_order,
                                      regularization=detrend_regularization,
                                      alpha=detrend_alpha)

    if filter_lowcut == 0:
        actual_filter_lowcut = None
    else:
        actual_filter_lowcut = filter_lowcut

    if filter_type != "None":
        distorted = nk.signal_filter(signal=distorted,
                                     sampling_rate=sampling_rate,
                                     lowcut=actual_filter_lowcut,
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
        rate = nk.rsp_rate(peaks=extrema_signal, sampling_rate=sampling_rate)["RSP_Rate"]
    except ValueError:
        rate = np.full(len(distorted), np.nan)

    info["Detrend_Method"] = [detrend_method]
    info["Detrend_Order"] = [detrend_order]
    info["Detrend_Regularization"] = [detrend_regularization]
    info["Detrend_Alpha"] = [detrend_alpha]
    info["Detrend_Position"] = [detrend_position]

    info["Filter_Method"] = [filter_type]
    if filter_type == "Butterworth":
        info["Filter_Type"] = [filter_type + "_" + str(filter_order)]
    else:
        info["Filter_Type"] = [filter_type]
    info["Filter_Order"] = [filter_order]
    info["Filter_Low"] = [filter_lowcut]
    info["Filter_High"] = [filter_highcut]
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
rsp, info = rsp_generate(duration=60, sampling_rate=200, method="Simple")
data = pd.DataFrame({"Original": data})
for noise_frequency in np.linspace(0.5, 95, 10):

    distorted, info = rsp_distord(rsp, info, noise_amplitude=noise_amplitude, noise_frequency=noise_frequency)
    data["Distorted_" + str(round(noise_frequency, 2))] = distorted





# =============================================================================
# RSP - Filter
# =============================================================================
#all_data = []
#for noise_amplitude in np.linspace(0.01, 1, 10):
#    print("---")
#    print(noise_amplitude*100)
#    print("---")
#    for noise_frequency in np.linspace(0.5, 95, 10):
##        print("%.2f" %(noise_frequency))
#        rsp, info = rsp_generate(duration=60, sampling_rate=200, respiratory_rate=15, method="Simple")
#        distorted, info = rsp_distord(rsp, info, noise_amplitude=noise_amplitude, noise_frequency=noise_frequency)
#
#        rate, info, cleaned = rsp_custom_process(distorted, info, detrend_position="None", filter_type="None")
#        data = rsp_quality(rate, info, cleaned, rsp)
#        all_data += [data]
#
#        for filter_highcut in np.linspace(35, 60, 10):
#
#            for filter_type in ["Butterworth", "FIR"]:
#                if filter_type == "FIR":
#                    rate, info, cleaned = rsp_custom_process(distorted, info, detrend_position="None", filter_type=filter_type, filter_highcut=filter_highcut)
#                    data = rsp_quality(rate, info, cleaned, rsp)
#                    all_data += [data]
#                else:
#                    for filter_order in range(10):
#                        rate, info, cleaned = rsp_custom_process(distorted, info, detrend_position="None", filter_type=filter_type, filter_order=filter_order, filter_highcut=filter_highcut)
#                        data = rsp_quality(rate, info, cleaned, rsp)
#                        all_data += [data]
#
#    data = pd.concat(all_data)
#    data.to_csv("data_RSP_filtering.csv")





# =============================================================================
# RSP - Detrending Order
# =============================================================================
#all_data = []
#for noise_amplitude in np.linspace(0.01, 1, 10):
#    print("---")
#    print(noise_amplitude*100)
#    print("---")
#    for noise_frequency in np.linspace(0.1, 100, 10):
##        print("%.2f" %(noise_frequency))
#        rsp, info = rsp_generate(duration=60, sampling_rate=100, respiratory_rate=15, method="Simple")
#        distorted, info = rsp_distord(rsp, info, noise_amplitude=noise_amplitude, noise_frequency=noise_frequency)
#
#        # None
#        rate, info = rsp_custom_process(distorted, info, detrend_position="None", filter_apply=False)
#        data = rsp_quality(rate, info, cleaned)
#        data["Detrend_Parameter"] = np.nan
#        all_data += [data]
#
#        detrend_order = np.arange(0, 10)
#        detrend_alpha = np.linspace(0.25, 0.95, 10)
#        detrend_regularization = np.linspace(100, 800, 10)
#        for i in range(10):
#
#            # Polynomial
#            rate, info, cleaned = rsp_custom_process(distorted, info, detrend_method="polynomial", detrend_order=detrend_order[i], filter_apply=False)
#            data = rsp_quality(rate, info)
#            data["Detrend_Parameter"] = i
#            all_data += [data]
#
#            # tarvainen2002
##            rate, info, cleaned = rsp_custom_process(distorted, info, detrend_method="tarvainen2002", detrend_regularization=detrend_regularization[i], filter_apply=False)
##            data = rsp_quality(rate, info, cleaned)
##            data["Detrend_Parameter"] = i
##            all_data += [data]
#
#            # Loess
##            rate, info, cleaned = rsp_custom_process(distorted, info, detrend_method="loess", detrend_alpha=detrend_alpha[i], filter_apply=False)
##            data = rsp_quality(rate, info, cleaned)
##            data["Detrend_Parameter"] = i
##            all_data += [data]
#
#    data = pd.concat(all_data)
#    data.to_csv("data_DetrendingOrder.csv")


# =============================================================================
# RSP - Filter
# =============================================================================
#all_data = []
#for noise_amplitude in np.linspace(0.01, 1, 5):
#    print("---")
#    print(noise_amplitude*100)
#    print("---")
#    for noise_frequency in np.linspace(1, 150, 5):
#        print("%.2f" %(noise_frequency/150*100))
#        for simulation in ["Complex"]:
#            for detrend_position in ["First", "Second", "None"]:
#                for detrend_order in [0, 1, 2, 3, 4, 5, 6]:
#                    for filter_order in [1, 2, 3, 4, 5, 6]:
#                        for filter_lowcut in [0, 0.05, 0.1, 0.15, 0.2]:
#                            for filter_highcut in [3, 2, 1, 0.35, 0.25]:
#                                rsp, info = rsp_generate(duration=120, sampling_rate=1000, respiratory_rate=15, method=simulation)
#                                distorted, info = rsp_distord(rsp, info, noise_amplitude=noise_amplitude, noise_frequency=noise_frequency)
#                                rate, info = rsp_custom_process(distorted, info,
#                                                                detrend_position=detrend_position,
#                                                                detrend_order=detrend_order,
#                                                                filter_order=filter_order,
#                                                                filter_lowcut=filter_lowcut,
#                                                                filter_highcut=filter_highcut)
#                                data = rsp_quality(rate, info)
#                                all_data += [data]
#    data = pd.concat(all_data)
#    data.to_csv("data.csv")
#
## Check
#fig, axes = plt.subplots(nrows=2, ncols=2)
#
#data.plot.scatter(x="Noise_Amplitude", y="Difference_Mean", color='r', ax=axes[0,0])
#data.plot.scatter(x="Noise_Amplitude", y="Difference_SD", color='r', ax=axes[1,0])
#data.plot.scatter(x="Noise_Frequency", y="Difference_Mean", color='g', ax=axes[0,1])
#data.plot.scatter(x="Noise_Frequency", y="Difference_SD", color='g', ax=axes[1,1])

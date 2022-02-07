# -*- coding: utf-8 -*-
# import numpy as np
# import pandas as pd


# def mne_channel_extract(raw):
#     """1/f Neural Noise

#     Extract parameters related to the 1/f structure of the EEG power spectrum.

#     Parameters
#     ----------
#     raw : mne.io.Raw
#         Raw EEG data.

#     Returns
#     ----------
#     DataFrame
#         A DataFrame or Series containing the channel(s).

#     Example
#     ----------
#     >>> import neurokit2 as nk
#     >>> import mne
#     >>>
#     >>> raw = nk.mne_data("raw")
#     >>>
#     """
#     import mne

#     import neurokit2 as nk

#     raw = nk.mne_data("raw")
#     raw.plot_psd(fmin=0, fmax=40.0, picks=["EEG 050"])

#     channel = nk.mne_channel_extract(raw, what=["EEG 050"]).values
#     psd = nk.signal_psd(
#         channel, sampling_rate=raw.info["sfreq"], show=True, max_frequency=40, method="multitapers"
#     )
#     plt.loglog(psd["Frequency"], psd["Power"])
#     plt.plot(psd["Frequency"], np.log(psd["Power"]))

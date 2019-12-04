# -*- coding: utf-8 -*-
import os
import datetime
import bioread

import pandas as pd
import numpy as np


def read_acqknowledge(filename, sampling_rate="max", resample_method="interpolation", impute_missing=True):
    """Read and format a BIOPAC's AcqKnowledge file into a pandas' dataframe.

    The function outputs both the dataframe and the sampling rate (encoded within the AcqKnowledge) file.

    Parameters
    ----------
    filename :  str
        Filename (with or without the extension) of a BIOPAC's AcqKnowledge file.
    sampling_rate : int
        Sampling rate (in Hz, i.e., samples/second). Since an AcqKnowledge file can contain signals recorded at different rates, harmonization is necessary in order to convert it to a DataFrame. Thus, if `sampling_rate` is set to 'max' (default), will keep the maximum recorded sampling rate and upsample the channels with lower rate if necessary (using the `signal_resample()` function). If the sampling rate is set to a given value, will resample the signals to the desired value. Note that the value of the sampling rate is outputted along with the data.
    resample_method : str
        Method of resampling (see `signal_resample()`). Can be 'interpolation' (default) or 'FFT' for the Fourier method. FFT is accurate (if the signal is periodic), but slower compared to interpolation.
    impute_missing : bool
        Sometimes, due to connections issues, the signal has some holes (short periods without signal). If 'impute_missing' is True, will automatically fill the signal interruptions using a backfill method (using the value to come).

    Returns
    ----------
    df, sampling rate: DataFrame, int
        The AcqKnowledge file converted to a dataframe and its sampling rate.

    See Also
    --------
    signal_resample

    Example
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> data, sampling_rate = nk.read_acqknowledge('file.acq')
    """



    # Check filename
    if ".acq" not in filename:
        filename += ".acq"

    if os.path.exists(filename) is False:
        raise ValueError("NeuroKit error: read_acqknowledge(): couldn't find the following file: " + filename)


    # Read file
    file = bioread.read(filename)
    bioread.read_file(filename)


    # Get the channel frequencies
    freq_list = []
    for channel in file.named_channels:
        freq_list.append(file.named_channels[channel].samples_per_second)

    # Get data with max frequency and the others
    data = {}
    data_else = {}
    for channel in file.named_channels:
        if file.named_channels[channel].samples_per_second == max(freq_list):
            data[channel] = file.named_channels[channel].data
        else:
            data_else[channel] = file.named_channels[channel].data

    # Create index
    time = []
    beginning_date = creation_date - datetime.timedelta(0, max(file.time_index))
    for timestamps in file.time_index:
        time.append(beginning_date + datetime.timedelta(0, timestamps))
    df = pd.DataFrame(data, index=time)





    # max frequency must be 1000
    if len(data_else.keys()) > 0:  # if not empty
        for channel in data_else:
            channel_frequency = file.named_channels[channel].samples_per_second
            serie = data_else[channel]
            index = list(np.arange(0, max(file.time_index), 1/channel_frequency))
            index = index[:len(serie)]

            # Create index
            time = []
            for timestamps in index:
                time.append(beginning_date + datetime.timedelta(0, timestamps))
            data_else[channel] = pd.Series(serie, index=time)
        df2 = pd.DataFrame(data_else)

    # Create resampling factor
    if sampling_rate == "max":
        sampling_rate = max(freq_list)

    try:
        resampling_factor = str(int(1000/sampling_rate)) + "L"
    except TypeError:
        print("NeuroKit Warning: read_acqknowledge(): sampling_rate must be either num or 'max'. Setting to 'max'.")
        sampling_rate = max(freq_list)
        resampling_factor = str(int(1000/sampling_rate)) + "L"


    # Resample
    if resampling_method not in ["mean", "bfill", "pad"]:
        print("NeuroKit Warning: read_acqknowledge(): resampling_factor must be 'mean', 'bfill' or 'pad'. Setting to 'pad'.")
        resampling_method = 'pad'

    if resampling_method == "mean":
        if len(data_else.keys()) > 0:
            df2 = df2.resample(resampling_factor).mean()
        if int(sampling_rate) != int(max(freq_list)):
            df = df.resample(resampling_factor).mean()
    if resampling_method == "bfill":
        if len(data_else.keys()) > 0:
            df2 = df2.resample(resampling_factor).bfill()
        if int(sampling_rate) != int(max(freq_list)):
            df = df.resample(resampling_factor).bfill()
    if resampling_method == "pad":
        if len(data_else.keys()) > 0:
            df2 = df2.resample(resampling_factor).pad()
        if int(sampling_rate) != int(max(freq_list)):
            df = df.resample(resampling_factor).pad()



    # Join dataframes
    if len(data_else.keys()) > 0:
        df = pd.concat([df, df2], 1)

    if index == "range":
        df = df.reset_index()

    # Fill signal interruptions
    if impute_missing is True:
        df = df.fillna(method="backfill")

    # Final dataframe
    return(df, sampling_rate)
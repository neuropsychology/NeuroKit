# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import bioread

from .find_creation_date import find_creation_date


def read_acqknowledge(filename, path="", index="datetime", sampling_rate="max", resampling_method="pad", fill_interruptions=True):
    """
    Read and Format a BIOPAC's AcqKnowledge file into a pandas' dataframe.

    Parameters
    ----------
    filename :  str
        Filename (with or without the extension) of a BIOPAC's AcqKnowledge file.
    path : str
        Data directory.
    index : str
        How to index the dataframe. "datetime" for aproximate datetime (based on the file creation/change) and "range" for a simple range index.
    sampling_rate : int
        Final sampling rate (samples/second).
    resampling_method : str
        The resampling method: "mean", "pad" or "bfill",
    fill_interruptions : bool
        Automatically fill the eventual signal interruptions using a backfill method.
    Returns
    ----------
    df: pandas.DataFrame()
        The AcqKnowledge file converted to a dataframe.


    Example
    ----------
    >>> import neurokit as nk
    >>>
    >>> df = nk.read_acqknowledge('file.acq')
    """



    # Check path
    file = path + filename
    if ".acq" not in file:
        file += ".acq"
    if os.path.exists(file) is False:
        print("NeuroKit Error: read_acqknowledge(): couldn't find the following file: " + filename)
        return()

    # Convert creation date
    creation_date = find_creation_date(file)
    creation_date = datetime.datetime.fromtimestamp(creation_date)


    # Read file
    file = bioread.read(file)


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
    if fill_interruptions is True:
        df = df.fillna(method="backfill")

    # Final dataframe
    return(df)
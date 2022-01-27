# -*- coding: utf-8 -*-
import json
import os

import numpy as np
import pandas as pd

from ..signal import signal_resample


def read_bitalino(
    filename,
    sampling_rate="max",
    resample_method="interpolation",
    events_annotation=False,
    events_annotation_directory=None,
):
    """Read and format a  OpenSignals file (e.g., from BITalino) into a pandas' dataframe.

    The function outputs both the dataframe and the sampling rate (retrieved from the
    OpenSignals file).

    Parameters
    ----------
    filename :  str
        Filename (with or without the extension) of an OpenSignals file (e.g., 'data.txt').
    sampling_rate : int
        Sampling rate (in Hz, i.e., samples/second). Defaults to the original sampling rate at which signals were
        sampled if set to "max". If the sampling rate is set to a given value, will resample
        the signals to the desired value. Note that the value of the sampling rate is outputted
        along with the data.
    resample_method : str
        Method of resampling (see `signal_resample()`).
    events_annotation : bool
        Defaults to False. If True, will read signal annotation events.
    events_annotation_directory : str
        If None (default), reads signal annotation events from the same location where the acquired file is stored. If not,
        specify the predefined OpenSignals (r)evolution folder directory of where the 'EventsAnnotation.txt' file is stored.

    Returns
    ----------
    df : DataFrame, dict
        The BITalino file as a pandas dataframe if one device was read, or a dictionary
        of pandas dataframes (one dataframe per device) if multiple devices are read.
    info : dict
        The metadata information containing the sensors, corresponding channel names, sampling rate, and the
        events annotation timings if `events_annotation` is True.

    See Also
    --------
    read_acqknowledge, signal_resample

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # data, sampling_rate = nk.read_bitalino("data.txt")
    """

    # read metadata
    with open(filename, "r") as f:

        if "OpenSignals" not in f.readline():  # read first line
            raise ValueError(
                "NeuroKit error: read_bitalino(): Text file is not in OpenSignals format."
            )

        metadata = json.loads(f.readline()[1:])  # read second line

    if len(list(metadata.keys())) == 1:
        return _read_bitalino_onedevice(
            filename,
            metadata,
            sampling_rate,
            resample_method,
            events_annotation,
            events_annotation_directory,
        )

    else:
        return _read_bitalino_multipledevice(
            filename,
            metadata,
            sampling_rate,
            resample_method,
            events_annotation,
            events_annotation_directory,
        )


# =============================================================================
# Convenience functions
# =============================================================================
def _read_bitalino_onedevice(
    filename,
    metadata,
    sampling_rate="max",
    resample_method="interpolation",
    events_annotation=False,
    events_annotation_directory=None,
):
    info = {}  # Initialize empty dict for storing

    # If only one device
    metadata = metadata[
        list(metadata.keys())[0]
    ]  # convert json header to dict (only select first device / MAC address)
    channels = np.arange(len(metadata["channels"])) + 5  # analog channels start from column 5

    data = pd.read_csv(filename, sep="\t", usecols=channels, header=None, comment="#")
    data.columns = metadata["sensor"]
    info["sensors"] = metadata["sensor"]
    info["channel_names"] = metadata["label"]

    # Adjust sampling rate
    if sampling_rate == "max":
        sampling_rate = metadata["sampling rate"]
    else:
        # resample
        colnames = list(data.columns)
        data, sampling_rate = _read_bitalino_resample(
            data,
            original_sampling_rate=metadata["sampling rate"],
            sampling_rate=sampling_rate,
            resample_method=resample_method,
        )
        data.columns = colnames
    info["sampling_rate"] = sampling_rate

    # Add manual events annotation
    if events_annotation:
        events = _read_bitalino_events_annotation(
            events_annotation_directory, info["channel_names"]
        )

        for event in events.keys():
            for chname in events[event].keys():
                # Initiate event columns in dataframe
                start = np.zeros(len(data))
                stop = np.zeros(len(data))

                # Convert timings to samples
                start_times = [int(i * sampling_rate) for i in events[event][chname]["start"]]
                stop_times = [int(i * sampling_rate) for i in events[event][chname]["stop"]]
                start[start_times] = 1
                stop[stop_times] = 1
                data[chname + "_" + event + "_start"] = start.astype(int)
                data[chname + "_" + event + "_stop"] = stop.astype(int)
        info["events annotation"] = events

    return data, info


def _read_bitalino_multipledevice(
    filename,
    metadata,
    sampling_rate="max",
    resample_method="interpolation",
    events_annotation=False,
    events_annotation_directory=None,
):
    info = {}  # Initialize empty dict for storing

    # Read from multiple devices
    devices = list(metadata.keys())
    data = {}
    for index, name in enumerate(devices):

        info[name] = {}
        channels = np.arange(len(metadata[name]["channels"])) + 5 + (5 * index) + (2 * index)
        # analog channels start from column 5 for each device

        df = pd.read_csv(filename, sep="\t", usecols=channels, header=None, comment="#")
        df.columns = metadata[name]["sensor"]
        info[name]["sensors"] = metadata[name]["sensor"]
        info[name]["channel_names"] = metadata[name]["label"]

        # Adjust sampling rate
        if sampling_rate == "max":
            sampling_rate = metadata[name]["sampling rate"]
        else:
            # resample
            colnames = list(df.columns)
            df, sampling_rate = _read_bitalino_resample(
                df,
                original_sampling_rate=metadata[name]["sampling rate"],
                resampling_rate=sampling_rate,
                resample_method=resample_method,
            )
            df.columns = colnames
        info[name]["sampling_rate"] = sampling_rate

        # Add manual events annotation
        if events_annotation:
            metaevents = _read_bitalino_events_annotation(
                events_annotation_directory, info[name]["channel_names"]
            )
            events = metaevents[name.replace(":", "")]

            for event in events.keys():
                for chname in events[event].keys():
                    # Initiate event columns in dataframe
                    start = np.zeros(len(df))
                    stop = np.zeros(len(df))

                    # Convert timings to samples
                    start_times = [int(i * sampling_rate) for i in events[event][chname]["start"]]
                    stop_times = [int(i * sampling_rate) for i in events[event][chname]["stop"]]
                    start[start_times] = 1
                    stop[stop_times] = 1
                    df[chname + "_" + event + "_start"] = start.astype(int)
                    df[chname + "_" + event + "_stop"] = stop.astype(int)
            info[name]["events annotation"] = events

    data[name] = df  # dict of dataframes for each device
    return data, info


# =============================================================================
# Internals
# =============================================================================
def _read_bitalino_resample(
    data, original_sampling_rate, resampling_rate, resample_method="interpolation"
):

    signals = pd.DataFrame()
    for i in data:
        signal = signal_resample(
            data[i],
            sampling_rate=original_sampling_rate,
            desired_sampling_rate=resampling_rate,
            method=resample_method,
        )
        signal = pd.Series(signal)
        signals = pd.concat([signals, signal], axis=1)
    data = signals.copy()

    return data, resampling_rate


def _read_bitalino_events_annotation(events_annotation_directory=None, channel_names=None):
    """Read events that are annotated during BITalino signal acquisition.
    Returns a dictionary containing the start and stop times (in seconds) in each channel detected per unique event
    (label) within each device."""

    # Get working directory of data file (assume events stored together in same folder)
    folder = os.listdir(events_annotation_directory)
    if len([i for i in folder if "_EventsAnnotation.txt" in i]) == 0:
        raise ValueError(
            "NeuroKit error: _read_bitalino_events_annotation(): No events annotation file found in the working directory. "
            + "Please specify `events_annotation_directory` argument to where the events annotation text file is stored."
        )
    else:
        events_file = [i for i in folder if "_EventsAnnotation.txt" in i][0]

    # read metadata
    with open(events_file, "r") as f:

        if "OpenSignals" not in f.readline():  # read first line
            raise ValueError(
                "NeuroKit error: read_bitalino(): Events text file is not in OpenSignals format."
            )

        eventdata = json.loads(f.readline()[1:])  # read second line

    df = pd.read_csv(events_file, sep="\t", header=None, comment="#").dropna(axis=1)
    df.columns = eventdata["columns"]["labels"]

    df = df[df["CHANNEL"].isin(channel_names)]  # read only from recorded channels

    # Initialize data
    metaevents = {}
    for device in np.unique(df["MAC"]):
        metaevents[device] = {}
        for key in np.unique(df["ID"]):
            key = "label" + str(key)
            metaevents[device][key] = {}

            device_data = df[df["MAC"] == device]
            for channel in np.unique(df["CHANNEL"]):
                metaevents[device][key][channel] = {}

                # Append data
                start = list(device_data[device_data["CHANNEL"] == channel]["START"])
                stop = list(device_data[device_data["CHANNEL"] == channel]["END"])
                metaevents[device][key][channel]["start"] = start
                metaevents[device][key][channel]["stop"] = stop

    return metaevents

# -*- coding: utf-8 -*-
import json

import numpy as np
import pandas as pd

from ..signal import signal_resample


def read_bitalino(filename, sampling_rate="max", resample_method="interpolation"):
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

    Returns
    ----------
    df : DataFrame
        The BITalino file as a pandas dataframe.
    sampling rate: int
        The sampling rate at which the data is sampled.

    See Also
    --------
    read_acqknowledge, signal_resample

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> #data, sampling_rate = nk.read_bitalino("data.txt")
    """
    # read metadata
    with open(filename, "r") as f:

        if "OpenSignals" not in f.readline():  # read first line
            raise ValueError(
                "NeuroKit error: read_bitalino(): Text file is not in OpenSignals format."
            )

        metadata = json.loads(f.readline()[1:])  # read second line

    metadata = metadata[
        list(metadata.keys())[0]
    ]  # convert json header to dict (only select first device / MAC address)
    channels = np.arange(len(metadata["channels"])) + 5  # analog channels start from column 5

    # Get desired frequency and produce output accordingly
    data = pd.read_csv(filename, sep="\t", usecols=channels, header=None, comment="#")

    if sampling_rate == "max":
        sampling_rate = metadata["sampling rate"]
    else:
        signals = pd.DataFrame()
        for i in data:
            signal = signal_resample(
                data[i],
                sampling_rate=metadata["sampling rate"],
                desired_sampling_rate=sampling_rate,
                method=resample_method,
            )
            signal = pd.Series(signal)
            signals = pd.concat([signals, signal], axis=1)
        data = signals.copy()

    data.columns = metadata["sensor"]

    return data, sampling_rate

# -*- coding: utf-8 -*-
import json

import numpy as np
import pandas as pd


def read_bitalino(filename):
    """Read and format a  OpenSignals file (e.g., from BITalino) into a pandas' dataframe.

    The function outputs both the dataframe and the sampling rate (retrieved from the
    OpenSignals file).

    Parameters
    ----------
    filename :  str
        Filename (with or without the extension) of an OpenSignals file (e.g., 'data.txt').

    Returns
    ----------
    df : DataFrame
        The AcqKnowledge file as a pandas dataframe.
    sampling rate: int
        The sampling rate at which the data is sampled.

    See Also
    --------
    read_acqknowledge

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
    sampling_rate = metadata["sampling rate"]
    channels = np.arange(len(metadata["channels"])) + 5  # analog channels start from column 5

    # Read data
    data = pd.read_csv(filename, sep="\t", usecols=channels, header=None, comment="#")
    data.columns = metadata["sensor"]

    return data, sampling_rate

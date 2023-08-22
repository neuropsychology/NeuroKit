# -*- coding: utf-8 -*-
import json
import os
from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning


def read_bitalino(filename):
    """**Read an OpenSignals file (from BITalino)**

    Reads and loads a BITalino file into a Pandas DataFrame.
    The function outputs both the dataframe and the information (such as the sampling rate)
    retrieved from the OpenSignals file.

    Parameters
    ----------
    filename :  str
        Path (with or without the extension) of an OpenSignals file (e.g., ``"data.txt"``).

    Returns
    ----------
    df : DataFrame, dict
        The BITalino file as a pandas dataframe if one device was read, or a dictionary
        of pandas dataframes (one dataframe per device) if multiple devices are read.
    info : dict
        The metadata information containing the sensors, corresponding channel names, sampling
        rate, and the events annotation timings if ``events_annotation`` is ``True``.

    See Also
    --------
    .read_acqknowledge, .signal_resample

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # data, info = nk.read_bitalino("data.txt")
      # sampling_rate = info["sampling_rate"]
    """

    # Read metadata
    # -------------------------------------------------------------------------
    with open(filename, "r") as f:
        lines = f.readlines()
        if "OpenSignals" not in lines[0]:
            raise ValueError("Text file is not in OpenSignals format.")
        metadata = json.loads(lines[1][1:])  # read second line + skip '#'

    # Remove ":"
    metadata = {k.replace(":", ""): metadata[k] for k in metadata.keys()}

    # Try find events annotations
    # -------------------------------------------------------------------------
    annotations = _read_bitalino_annotations(filename)
    if annotations is not None:
        for k in annotations.keys():
            if k in metadata.keys():
                metadata[k]["Annotations"] = annotations[k]
            else:
                warn(
                    f"Device {k} not found in metadata ({metadata.keys()})."
                    + " Something might be wrong.",
                    category=NeuroKitWarning,
                )

    # Read data
    # -------------------------------------------------------------------------
    data = {k: None for k in metadata.keys()}
    raw = pd.read_csv(filename, sep="\t", header=None, comment="#")

    # Read file for each device
    for i, k in enumerate(metadata.keys()):
        # Select right columns
        ch = np.array(metadata[k]["column"])
        data[k] = raw.iloc[:, i * len(ch) : (i + 1) * len(ch)]

        for j, s in enumerate(metadata[k]["label"]):
            ch[ch == s] = metadata[k]["sensor"][j]
        data[k].columns = ch

        # Add annotations
        if "Annotations" in metadata[k].keys():
            sr = metadata[k]["sampling rate"]
            data[k]["Events"] = 0
            annot = metadata[k]["Annotations"]
            annot = annot[annot["CHANNEL"].isin(metadata[k]["label"])]
            annot = annot.drop_duplicates(["START", "END"])
            for _, row in annot.iterrows():
                data[k]["Events"][int(row["START"] * sr) : int(row["END"] * sr) + 1] = 1

        # Format metadata names
        metadata[k] = {x.replace(" ", "_"): v for x, v in metadata[k].items()}

    # If only one device is detected, extract from dict
    if i == 0:
        data = data[k]
        metadata = metadata[k]
    return data, metadata


# =============================================================================
# Internals
# =============================================================================
def _read_bitalino_annotations(filename):
    """Read events that are annotated during BITalino signal acquisition.

    Returns a dictionary containing the start and stop times (in seconds) in each channel detected
    per unique event (label) within each device."""

    file = filename.replace(".txt", "_EventsAnnotation.txt")
    if os.path.isfile(file) is False:
        return None

    with open(file, "r") as f:
        lines = f.readlines()
        if "OpenSignals" not in lines[0]:
            raise ValueError("Text file is not in OpenSignals format.")
        metadata = json.loads(lines[1][1:])  # read second line + skip '#'
        data = pd.read_csv(file, sep="\t", header=None, comment="#")
        data = data.dropna(axis=1, how="all")
        data.columns = metadata["columns"]["labels"]

    return {k: data[data["MAC"] == k] for k in data["MAC"].unique()}

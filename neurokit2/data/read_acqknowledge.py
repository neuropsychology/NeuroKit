# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd

from ..signal import signal_resample


def read_acqknowledge(filename, sampling_rate="max", resample_method="interpolation", impute_missing=True):
    """Read and format a BIOPAC's AcqKnowledge file into a pandas' dataframe.

    The function outputs both the dataframe and the sampling rate (encoded within the
    AcqKnowledge) file.

    Parameters
    ----------
    filename :  str
        Filename (with or without the extension) of a BIOPAC's AcqKnowledge file.
    sampling_rate : int
        Sampling rate (in Hz, i.e., samples/second). Since an AcqKnowledge file can contain
        signals recorded at different rates, harmonization is necessary in order to convert it
        to a DataFrame. Thus, if `sampling_rate` is set to 'max' (default), will keep the maximum
        recorded sampling rate and upsample the channels with lower rate if necessary (using the
        `signal_resample()` function). If the sampling rate is set to a given value, will resample
        the signals to the desired value. Note that the value of the sampling rate is outputted
        along with the data.
    resample_method : str
        Method of resampling (see `signal_resample()`).
    impute_missing : bool
        Sometimes, due to connections issues, the signal has some holes (short periods without
        signal). If 'impute_missing' is True, will automatically fill the signal interruptions
        using padding.

    Returns
    ----------
    df : DataFrame
        The AcqKnowledge file converted to a dataframe.
    sampling rate: int
        The AcqKnowledge file converted to its sampling rate.

    See Also
    --------
    signal_resample

    Example
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> data, sampling_rate = nk.read_acqknowledge('file.acq') #doctest: +SKIP

    """
    # Try loading bioread
    try:
        import bioread
    except ImportError:
        raise ImportError(
            "NeuroKit error: read_acqknowledge(): the 'bioread' module is required",
            " for this function to run. ",
            "Please install it first (`pip install bioread`).",
        )

    # Check filename
    if ".acq" not in filename:
        filename += ".acq"

    if os.path.exists(filename) is False:
        raise ValueError("NeuroKit error: read_acqknowledge(): couldn't" " find the following file: " + filename)

    # Read file
    file = bioread.read(filename)

    # Get desired frequency
    if sampling_rate == "max":
        freq_list = []
        for channel in file.named_channels:
            freq_list.append(file.named_channels[channel].samples_per_second)
        sampling_rate = np.max(freq_list)

    # Loop through channels
    data = {}
    for channel in file.named_channels:
        signal = np.array(file.named_channels[channel].data)

        # Fill signal interruptions
        if impute_missing is True and np.isnan(np.sum(signal)):
            signal = pd.Series(signal).fillna(method="pad").values

        # Resample if necessary
        if file.named_channels[channel].samples_per_second != sampling_rate:
            signal = signal_resample(
                signal,
                sampling_rate=file.named_channels[channel].samples_per_second,
                desired_sampling_rate=sampling_rate,
                method=resample_method,
            )
        data[channel] = signal

    # Sanitize lengths
    lengths = []
    for channel in data:
        lengths += [len(data[channel])]
    if len(set(lengths)) > 1:  # If different lengths
        length = pd.Series(lengths).mode()[0]  # Find most common (target length)
        for channel in data:
            if len(data[channel]) > length:
                data[channel] = data[channel][0:length]
            if len(data[channel]) < length:
                data[channel] = np.concatenate(
                    [data[channel], np.full((length - len(data[channel])), data[channel][-1])]
                )

    # Final dataframe
    df = pd.DataFrame(data)
    return df, sampling_rate

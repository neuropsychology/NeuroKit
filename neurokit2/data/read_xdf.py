# -*- coding: utf-8 -*-
import io
import urllib

import numpy as np
import pandas as pd
import requests


def read_xdf(
    filename: str, upsample: float = 2.0, fillmissing: float | None = None
) -> tuple[pd.DataFrame, dict]:
    """**Read and tidy an XDF file**

    Reads and tidies an XDF file with multiple streams into a Pandas DataFrame.
    The function outputs both the dataframe and the information (such as the sampling rate).

    Note that, as XDF can store streams with different sampling rates and different time stamps,
    **the function will resample all streams to 2 times (default) the highest sampling rate** (to
    minimize aliasing) and then interpolate based on an evenly spaced index. While this is generally safe, it
    may produce unexpected results, particularly if the original stream has large gaps in its time series.
    For more discussion, see `here <https://github.com/xdf-modules/pyxdf/pull/1>`_.

    The final upsampled sampling rate can be found in the ``info`` dictionary.

    .. note::

        This function requires the *pyxdf* module to be installed. You can install it with
        ``pip install pyxdf``.

    Parameters
    ----------
    filename :  str
        Path (with the extension) or URL pointing to an XDF file (e.g., ``"data.xdf"``).
    upsample : float
        Factor by which to upsample the data. Default is 2.0, which means that the data will be
        resampled to 2 times the highest sampling rate. You can increase that to further reduce
        edge-distortion, especially for high frequency signals like EEG. ``1.0`` disables upsampling
        (but not interpolation).
    fillmissing : float
        The maximum duration in seconds of missing data to fill. ``None`` (default) will
        interpolate all missing values and prevent issues with NaNs. However, it might be important
        to keep the missing intervals (e.g., ``fillmissing=1`` to keep interruptions of more than
        1 s) typically corresponding to signal loss or streaming interruptions and exclude them
        from further analysis.

    Returns
    ----------
    df : DataFrame
        The XDF data as a pandas dataframe. If multiple streams are read,
        they will be merged into one dataframe.
    info : dict
        The metadata information containing the sampling rate(s).

    See Also
    --------
    .read_bitalino, .signal_resample

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # data, info = nk.read_xdf("data.xdf")
      # sampling_rate = info["sampling_rate"]

    """
    try:
        import pyxdf
    except ImportError as e:
        raise ImportError(
            "The 'pyxdf' module is required for this function to run. ",
            "Please install it first (`pip install pyxdf`).",
        ) from e

    # Load file
    # if filename is a URL, stream bytes from file
    if urllib.parse.urlparse(filename).scheme != "":
        try:
            req = requests.get(filename, stream=True, timeout=10)
            req.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            req.raw.decode_content = True
            filename = io.BytesIO(req.content)
        except requests.exceptions.RequestException as e:
            raise IOError(f"Failed to read XDF file from URL: {filename}") from e

    streams, header = pyxdf.load_xdf(filename)

    # Get smaller time stamp to later use as offset (zero point)
    min_ts = min(min(s["time_stamps"]) for s in streams)

    # Loop through all the streams and convert to dataframes
    dfs = []
    for stream in streams:
        # Get columns names and make dataframe
        channels_info = stream["info"]["desc"][0]["channels"][0]["channel"]
        cols = [channels_info[i]["label"][0] for i in range(len(channels_info))]
        dat = pd.DataFrame(stream["time_series"], columns=cols)

        # Special treatment for some devices
        if stream["info"]["name"][0] == "Muse":
            # Rename GYRO channels
            if stream["info"]["type"][0] == "GYRO":
                dat = dat.rename(columns={"X": "GYRO_X", "Y": "GYRO_Y", "Z": "GYRO_Z"})
                # Compute movement
                dat["GYRO"] = np.sqrt(
                    dat["GYRO_X"] ** 2 + dat["GYRO_Y"] ** 2 + dat["GYRO_Z"] ** 2
                )

            if stream["info"]["type"][0] == "ACC":
                dat = dat.rename(columns={"X": "ACC_X", "Y": "ACC_Y", "Z": "ACC_Z"})
                # Compute acceleration
                dat["ACC"] = np.sqrt(
                    dat["ACC_X"] ** 2 + dat["ACC_Y"] ** 2 + dat["ACC_Z"] ** 2
                )

            # Muse - PPG data has three channels: ambient, infrared, red
            if stream["info"]["type"][0] == "PPG":
                dat = dat.rename(
                    columns={"PPG1": "LUX", "PPG2": "PPG", "PPG3": "RED", "IR": "PPG"}
                )
                # Zeros suggest interruptions, better to replace with NaNs (I think?)
                dat["PPG"] = dat["PPG"].replace(0, value=np.nan)
                dat["LUX"] = dat["LUX"].replace(0, value=np.nan)

        # Get time stamps and offset from minimum time stamp
        dat.index = pd.to_datetime(stream["time_stamps"] - min_ts, unit="s")
        dfs.append(dat)

    # Store info of each stream ----------------------------------------------------------------

    # Store metadata
    info = {
        "sampling_rates_original": [
            float(s["info"]["nominal_srate"][0]) for s in streams
        ],
        "sampling_rates_effective": [
            float(s["info"]["effective_srate"]) for s in streams
        ],
        "datetime": header["info"]["datetime"][0],
        "data": dfs,
    }

    # Synchronize ------------------------------------------------------------------------------
    # Merge all dataframes by timestamps
    # Note: this is a critical steps, as it inserts timestamps and makes it non-evenly spread
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = pd.merge(df, dfs[i], how="outer", left_index=True, right_index=True)
    df = df.sort_index()

    # Resample and Interpolate -----------------------------------------------------------------
    # Final sampling rate will be 2 times the maximum sampling rate
    # (to minimize aliasing during interpolation)
    info["sampling_rate"] = int(np.max(info["sampling_rates_original"]) * upsample)
    if fillmissing is not None:
        fillmissing = int(info["sampling_rate"] * fillmissing)

    # Create new index with evenly spaced timestamps
    idx = pd.date_range(
        df.index.min(), df.index.max(), freq=str(1000 / info["sampling_rate"]) + "ms"
    )
    # https://stackoverflow.com/questions/47148446/pandas-resample-interpolate-is-producing-nans
    df = (
        df.reindex(df.index.union(idx))
        .interpolate(method="index", limit=fillmissing)
        .reindex(idx)
    )

    return df, info

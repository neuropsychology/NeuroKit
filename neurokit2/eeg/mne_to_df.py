# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def mne_to_df(eeg):
    """**Conversion from MNE to dataframes**

    Convert MNE objects to dataframe or dict of dataframes.

    Parameters
    ----------
    eeg : Union[mne.io.Raw, mne.Epochs]
        Raw or Epochs M/EEG data from MNE.

    See Also
    --------
    .mne_to_dict

    Returns
    ----------
    DataFrame
        A DataFrame containing all epochs identifiable by the ``"Epoch"`` column, which time axis
        is stored in the ``"Time"`` column.

    Examples
    ---------
    **Raw objects**

    .. ipython:: python

      import neurokit2 as nk

      # Download MNE Raw object
      eeg = nk.mne_data("filt-0-40_raw")
      nk.mne_to_df(eeg)

    **Epoch objects**

    .. ipython:: python

      # Download MNE Epochs object
      eeg = nk.mne_data("epochs")
      nk.mne_to_df(eeg)

    **Evoked objects**

    .. ipython:: python

      # Download MNE Evoked object
      eeg = nk.mne_data("evoked")
      nk.mne_to_df(eeg)

    """
    return _mne_convert(eeg, to_what="df")


# Dict
def mne_to_dict(eeg):
    """**Convert MNE Raw or Epochs object to a dictionary**

    Parameters
    ----------
    eeg : Union[mne.io.Raw, mne.Epochs]
        Raw or Epochs M/EEG data from MNE.

    See Also
    --------
    mne_to_df

    Returns
    ----------
    dict
        A dict containing all epochs identifiable by the 'Epoch' column, which time axis
        is stored in the 'Time' column.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk
      import mne

      # Raw objects
      eeg = nk.mne_data("filt-0-40_raw")
      eeg_dict = nk.mne_to_dict(eeg)

      # Print function result summary
      eeg_dict_view = {k: f"Signal with length: {len(v)}" for k, v in eeg_dict.items()}
      eeg_dict_view


      # Epochs objects
      eeg = nk.mne_data("epochs")
      eeg_epoch_dict = nk.mne_to_dict(eeg)

      # Print function result summary
      list(eeg_epoch_dict.items())[:2]

      # Evoked objects
      eeg = nk.mne_data("evoked")
      eeg_evoked_dict = nk.mne_to_dict(eeg)

      # Print function result summary
      eeg_evoked_dict

    """
    return _mne_convert(eeg, to_what="dict")


# =============================================================================
# Main function
# =============================================================================
def _mne_convert(eeg, to_what="df"):
    # Try loading mne
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "NeuroKit error: eeg_add_channel(): the 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        ) from e

    old_verbosity_level = mne.set_log_level(verbose="WARNING", return_old_level=True)

    # If raw object
    if isinstance(eeg, (mne.io.Raw, mne.io.RawArray)):
        data = eeg.to_data_frame(time_format=None)
        data = data.rename(columns={"time": "Time"})
        if to_what == "dict":
            data = data.to_dict(orient="list")

    # If epoch object
    elif isinstance(eeg, mne.Epochs):
        data = eeg.to_data_frame(time_format=None)
        data = data.rename(columns={"time": "Time", "condition": "Condition", "epoch": "Epoch"})
        if to_what == "dict":
            out = {}
            for epoch in data["Epoch"].unique():
                out[epoch] = data[data["Epoch"] == epoch]
            data = out

    # If dataframe, skip and return
    elif isinstance(eeg, pd.DataFrame):
        data = eeg
        if to_what == "dict":
            data = data.to_dict(orient="list")

    # If dataframe, skip and return
    elif isinstance(eeg, np.ndarray):
        data = pd.DataFrame(eeg)
        if to_what == "dict":
            data = data.to_dict(orient="list")

    # it might be an evoked object
    else:
        dfs = []
        for i, evoked in enumerate(eeg):
            data = evoked.to_data_frame(time_format=None)
            data = data.rename(columns={"time": "Time"})
            data.insert(1, "Condition", evoked.comment)
            dfs.append(data)
        data = pd.concat(dfs, axis=0)
        if to_what == "dict":
            out = {}
            for condition in data["Condition"].unique():
                out[condition] = data[data["Condition"] == condition]
            data = out

    mne.set_log_level(old_verbosity_level)
    return data

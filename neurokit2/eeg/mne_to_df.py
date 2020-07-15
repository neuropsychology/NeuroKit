# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def mne_to_df(eeg):
    """Convert mne Raw or Epochs object to dataframe or dict of dataframes.

    Parameters
    ----------
    eeg : Union[mne.io.Raw, mne.Epochs]
        Raw or Epochs M/EEG data from MNE.

    Returns
    ----------
    DataFrame
        A DataFrame containing all epochs identifiable by the 'Label' column, which time axis
        is stored in the 'Time' column.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> eeg = nk.mne_data("filt-0-40_raw")
    >>> eeg = nk.eeg_rereference(eeg, 'average')
    >>>
    >>> gfp = nk.eeg_gfp(eeg)
    >>> gfp_z = nk.eeg_gfp(eeg, normalize=True)
    >>> gfp_zr = nk.eeg_gfp(eeg, normalize=True, robust=True)
    >>> gfp_s = nk.eeg_gfp(eeg, smooth=0.05)
    >>> nk.signal_plot([gfp[0:500], gfp_z[0:500], gfp_zr[0:500], gfp_s[0:500]], standardize=True)

    """
    # Try loading mne
    try:
        import mne
    except ImportError:
        raise ImportError(
            "NeuroKit error: eeg_add_channel(): the 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        )

    # If epoch object
    if isinstance(eeg, mne.Epochs):
        data = _mne_to_df_epochs(eeg)

    # If raw object
    elif isinstance(eeg, mne.io.Raw):
        data = _mne_to_df_raw(eeg)

    # If array or dataframe, skip and return
    elif isinstance(eeg, (pd.DataFrame, np.ndarray)):
        return data

    # it might be an evoked object
    else:
        data = _mne_to_df_evoked(eeg)

    return data


def mne_to_dict(eeg):
    """Convert MNE Raw or Epochs object to a dictionnary.

    Parameters
    ----------
    eeg : Union[mne.io.Raw, mne.Epochs]
        Raw or Epochs M/EEG data from MNE.

    Returns
    ----------
    DataFrame
        A DataFrame containing all epochs identifiable by the 'Label' column, which time axis
        is stored in the 'Time' column.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>> import mne
    >>>
    >>> raw = nk.mne_data()
    >>>
    >>> nk.mne_to_dict(raw)  # doctest: +SKIP

    """
    # Try loading mne
    try:
        import mne
    except ImportError:
        raise ImportError(
            "NeuroKit error: eeg_add_channel(): the 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        )

    # If epoch object
    if isinstance(eeg, mne.Epochs):
        data = _mne_to_dict_epochs(eeg)

    # If raw object
    elif isinstance(eeg, mne.io.Raw):
        data = _mne_to_dict_raw(eeg)

    # If array or dataframe, skip and return
    elif isinstance(eeg, (pd.DataFrame, np.ndarray)):
        return data

    # it might be an evoked object
    else:
        data = _mne_to_dict_evoked(eeg)

    return data


# =============================================================================
# epochs object
# =============================================================================
def _mne_to_dict_epochs(eeg):
    # Try loading mne
    try:
        import mne
    except ImportError:
        raise ImportError(
            "NeuroKit error: eeg_add_channel(): the 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        )

    data = {}

    old_verbosity_level = mne.set_log_level(verbose="WARNING", return_old_level=True)
    for i, dat in enumerate(eeg.get_data()):

        df = pd.DataFrame(dat.T)
        df.columns = eeg[i].ch_names
        df.index = eeg[i].times

        # Add info
        info = pd.DataFrame({"Label": [i] * len(df)})
        info["Condition"] = list(eeg[i].event_id.keys())[0]
        info["Time"] = eeg[i].times
        info.index = eeg[i].times

        data[i] = pd.concat([info, df], axis=1)

    mne.set_log_level(old_verbosity_level)
    return data


def _mne_to_df_epochs(eeg):
    data = _mne_to_dict_epochs(eeg)
    data = pd.concat(data)
    data = data.reset_index(drop=True)
    return data


# =============================================================================
# raw object
# =============================================================================
def _mne_to_dict_raw(eeg):
    data = _mne_to_df_raw(eeg)
    out = data.to_dict(orient="list")
    return out


def _mne_to_df_raw(eeg):
    data = eeg.get_data(verbose="WARNING").T
    data = pd.DataFrame(data)
    data.columns = eeg.ch_names
    data.index = eeg.times
    return data


# =============================================================================
# evoked object
# =============================================================================
def _mne_to_dict_evoked(eeg):
    # Try loading mne
    try:
        import mne
    except ImportError:
        raise ImportError(
            "NeuroKit error: eeg_add_channel(): the 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        )

    if not isinstance(eeg, list):
        eeg = [eeg]

    data = {}

    old_verbosity_level = mne.set_log_level(verbose="WARNING", return_old_level=True)
    for i, evoked in enumerate(eeg):
        df = evoked.to_data_frame()
        df.index = evoked.times

        # Add info
        info = pd.DataFrame({"Label": [i] * len(df)})
        info["Condition"] = evoked.comment
        info["Time"] = evoked.times
        info.index = evoked.times

        data[i] = pd.concat([info, df], axis=1)

    mne.set_log_level(old_verbosity_level)
    return data


def _mne_to_df_evoked(eeg):
    data = _mne_to_dict_evoked(eeg)
    data = pd.concat(data)
    data = data.reset_index(drop=True)
    return data

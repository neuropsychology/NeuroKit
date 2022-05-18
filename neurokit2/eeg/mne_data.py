# -*- coding: utf-8 -*-


def mne_data(what="raw", path=None):
    """**Access MNE Datasets**

    Utility function to easily access MNE datasets.

    Parameters
    -----------
    what : str
        Can be ``"raw"`` or ``"filt-0-40_raw"`` (a filtered version).
    path : str
        Defaults to ``None``, assuming that the MNE data folder already exists. If not,
        specify the directory to download the folder.

    Returns
    -------
    object
        The raw mne object.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      raw = nk.mne_data(what="raw")
      raw = nk.mne_data(what="epochs")

    """
    # Try loading mne
    try:
        import mne
    except ImportError:
        raise ImportError(
            "NeuroKit error: mne_data(): the 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        )

    old_verbosity_level = mne.set_log_level(verbose="WARNING", return_old_level=True)

    # Find path of mne data
    if path is None:
        try:
            path = str(mne.datasets.sample.data_path())
        except ValueError:
            raise ValueError(
                "NeuroKit error: the mne sample data folder does not exist. ",
                "Please specify a path to download the mne datasets.",
            )

    # Raw
    if what in ["raw", "filt-0-40_raw"]:
        path += "/MEG/sample/sample_audvis_" + what + ".fif"
        data = mne.io.read_raw_fif(path, preload=True)
        data = data.pick_types(meg=False, eeg=True)

    # Epochs
    elif what in ["epochs", "evoked"]:
        raw = mne.io.read_raw_fif(path + "/MEG/sample/sample_audvis_filt-0-40_raw.fif").pick_types(
            meg=False, eeg=True
        )

        events = mne.read_events(path + "/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif")
        event_id = {"audio/left": 1, "audio/right": 2, "visual/left": 3, "visual/right": 4}

        data = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5, baseline=(None, 0))

        if what in ["evoked"]:
            data = [data[name].average() for name in ("audio", "visual")]

    else:
        raise ValueError("NeuroKit error: mne_data(): the 'what' argument not recognized.")

    mne.set_log_level(old_verbosity_level)
    return data

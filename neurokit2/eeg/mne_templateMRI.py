import os


def mne_templateMRI(verbose="WARNING"):
    """**Return Path of MRI Template**

    This function is a helper that returns the path of the MRI template for adults (the ``src`` and
    the ``bem``) that is made available through ``"MNE"``. It downloads the data if need be. These
    templates can be used for EEG source reconstruction when no individual MRI is available.

    See https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html

    Parameters
    ----------
    verbose : str
        Verbosity level for MNE.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      src, bem = nk.mne_templateMRI()

    """
    # Try loading mne (requires also the 'pooch' package)
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "NeuroKit error: mne_templateMRI(): the 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        ) from e

    # Download fsaverage files
    fs_dir = mne.datasets.fetch_fsaverage(verbose=verbose)

    # The files live in:
    src = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
    return src, bem

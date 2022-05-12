import pandas as pd


def eeg_source_extract(stc, src, segmentation="PALS_B12_Lobes", verbose="WARNING", **kwargs):
    """**Extract the activity from an anatomical source**

    Returns a dataframe with the activity from each source in the segmentation.

    Parcellation models include:
    * 'aparc'
    * 'aparc.a2005s'
    * 'aparc.a2009s'
    * 'oasis.chubs'
    * 'PALS_B12_Brodmann'
    * 'PALS_B12_Lobes'
    * 'PALS_B12_OrbitoFrontal'
    * 'PALS_B12_Visuotopic'
    * 'Yeo2011_17Networks_N1000'
    * 'Yeo2011_7Networks_N1000'

    Parameters
    ----------
    stc : mne.SourceEstimate
        An SourceEstimate object as obtained by ``eeg_source()``.
    src : mne.SourceSpaces
        An SourceSpaces object as obtained by ``eeg_source()``.
    segmentation : str
        See above.
    verbose : str
        Verbosity level for MNE.
    **kwargs
        Other arguments to be passed to ``mne.extract_label_time_course()``.

    Examples
    ---------
    .. ipython:: python
      :verbatim:

      import neurokit2 as nk

      raw = nk.mne_data("filt-0-40_raw")

      src, bem = nk.mne_templateMRI()

      stc, src = nk.eeg_source(raw, src, bem)
      data = nk.eeg_source_extract(stc, src, segmentation="PALS_B12_Lobes")
      data.head()

    """
    # Try loading mne
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "NeuroKit error: mne_templateMRI(): the 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        ) from e

    # Find labels
    labels = mne.read_labels_from_annot(
        subject="fsaverage",
        parc=segmentation,
        subjects_dir=str(mne.datasets.sample.data_path()) + "/subjects",
        verbose=verbose,
    )

    # Filter empty ones
    labels = [lab for lab in labels if len(lab) > 0]
    # Filter Unknown ones
    labels = [lab for lab in labels if "?" not in lab.name]

    tcs = stc.extract_label_time_course(
        labels,
        src=src,
        verbose=verbose,
        **kwargs,
    )

    return pd.DataFrame(tcs.T, columns=[lab.name for lab in labels])

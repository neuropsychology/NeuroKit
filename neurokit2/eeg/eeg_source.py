def eeg_source(raw, src, bem, method="sLORETA", show=False, verbose="WARNING", **kwargs):
    """**Source Reconstruction for EEG data**

    Currently only for mne.Raw objects.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    src : mne.SourceSpace
        Source space. See :func:`mne_templateMRI()` to obtain it from an MRI template.
    bem : mne.Bem
        BEM model. See :func:`mne_templateMRI()` to obtain it from an MRI template.
    method : str
        Can be ``"sLORETA"``, ``"MNE"`` or ``"dSPM"``. See :func:`.mne.minimum_norm.apply_inverse_raw()`.
    show : bool
        If ``True``, shows the location of the electrodes on the head. See :func:`.mne.viz.plot_alignment()`.
    verbose : str
        Verbosity level for MNE.
    **kwargs
        Other arguments to be passed to ``mne.make_forward_solution()`` and
        ``mne.minimum_norm.make_inverse_operator()`` and ``mne.minimum_norm.apply_inverse_raw()``.

    See Also
    --------
    mne_templateMRI

    """
    # Try loading mne
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "NeuroKit error: mne_templateMRI(): the 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        ) from e

    if "trans" not in kwargs.keys():
        trans = "fsaverage"  # MNE has a built-in fsaverage transformation
    else:
        trans = kwargs.pop("trans")

    # Setup source space and compute forward
    fwd = mne.make_forward_solution(
        raw.info, trans=trans, src=src, bem=bem, verbose=verbose, **kwargs
    )

    # Get noise covariance matrix
    noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)

    # Get inverse solution
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        raw.info, fwd, noise_cov, verbose=verbose, **kwargs
    )
    src = inverse_operator["src"]

    snr = 1.0  # use smaller SNR for raw data
    # Compute inverse solution
    stc = mne.minimum_norm.apply_inverse_raw(
        raw,
        inverse_operator,
        lambda2=1.0 / snr ** 2,
        method=method,  # sLORETA method (could also be MNE or dSPM)
        verbose=verbose,
        **kwargs
    )

    # Plot
    if show is True:
        # Check that the locations of EEG electrodes is correct with respect to MRI
        # requires PySide2, ipyvtklink and mayavi
        mne.viz.plot_alignment(
            raw.info,
            src=src,
            eeg=["original", "projected"],
            trans=trans,
            mri_fiducials=True,
            dig="fiducials",
            verbose=verbose,
        )

    return stc, src

import numpy as np


def mne_crop(raw, tmin=0.0, tmax=None, include_tmax=True, smin=None, smax=None):
    """**Crop mne.Raw objects**

    This function is similar to ``raw.crop()`` (same arguments), but with a few critical differences:
    * It recreates a whole new Raw object, and as such drops all information pertaining to the
    original data (which MNE keeps, see https://github.com/mne-tools/mne-python/issues/9759).
    * There is the possibility of specifying directly the first and last samples (instead of in
    time unit).

    Parameters
    -----------
    raw : mne.io.Raw
        Raw EEG data.
    path : str
        Defaults to ``None``, assuming that the MNE data folder already exists. If not,
        specify the directory to download the folder.
    tmin : float
        See :func:`mne.Raw.crop()`.
    tmax : float
        See :func:`mne.Raw.crop()`.
    include_tmax : float
        See :func:`mne.Raw.crop()`.
    smin : int
        Cropping start in samples.
    samx : int
        Cropping end in samples.

    Returns
    -------
    mne.io.Raw
        a cropped mne.Raw object.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      raw = nk.mne_data(what="raw")
      raw_cropped = nk.mne_crop(raw, smin=200, smax=1200, include_tmax=False)
      len(raw_cropped)

    """
    # Try loading mne
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "NeuroKit error: eeg_channel_add(): the 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        ) from e

    # Convert time to samples
    if smin is None or smax is None:
        max_time = (raw.n_times - 1) / raw.info["sfreq"]
        if tmax is None:
            tmax = max_time

        if tmin > tmax:
            raise ValueError(f"tmin ({tmin}) must be less than tmax ({tmax})")
        if tmin < 0.0:
            raise ValueError(f"tmin ({tmin}) must be >= 0")
        elif tmax > max_time:
            raise ValueError(
                f"tmax ({tmax}) must be less than or equal to the max time ({max_time} sec)."
            )

        # Convert time to first and last samples
        new_smin, new_smax = np.where(
            _time_mask(raw.times, tmin, tmax, sfreq=raw.info["sfreq"], include_tmax=include_tmax)
        )[0][[0, -1]]

    if smin is None:
        smin = new_smin
    if smax is None:
        smax = new_smax
    if include_tmax:
        smax += 1

    # Re-create the Raw object (note that mne does smin : smin + 1)
    raw = mne.io.RawArray(raw._data[:, int(smin) : int(smax)].copy(), raw.info, verbose="WARNING")

    return raw


def _time_mask(times, tmin=None, tmax=None, sfreq=None, raise_error=True, include_tmax=True):
    """Copied from https://github.com/mne-tools/mne-python/mne/utils/numerics.py#L466."""
    orig_tmin = tmin
    orig_tmax = tmax
    tmin = -np.inf if tmin is None else tmin
    tmax = np.inf if tmax is None else tmax
    if not np.isfinite(tmin):
        tmin = times[0]
    if not np.isfinite(tmax):
        tmax = times[-1]
        include_tmax = True  # ignore this param when tmax is infinite
    if sfreq is not None:
        # Push to a bit past the nearest sample boundary first
        sfreq = float(sfreq)
        tmin = int(round(tmin * sfreq)) / sfreq - 0.5 / sfreq
        tmax = int(round(tmax * sfreq)) / sfreq
        tmax += (0.5 if include_tmax else -0.5) / sfreq
    else:
        assert include_tmax  # can only be used when sfreq is known
    if raise_error and tmin > tmax:
        raise ValueError(f"tmin ({orig_tmin}) must be less than or equal to tmax ({orig_tmax})")
    mask = times >= tmin
    mask &= times <= tmax
    if raise_error and not mask.any():
        extra = "" if include_tmax else "when include_tmax=False "
        raise ValueError(
            f"No samples remain when using tmin={orig_tmin} and tmax={orig_tmax} {extra}"
            "(original time bounds are [{times[0]}, {times[-1]}])"
        )
    return mask

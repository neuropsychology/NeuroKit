# -*- coding: utf-8 -*-
import pandas as pd

from ..signal import signal_rate
from ..signal.signal_formatpeaks import _signal_from_indices
from .eog_clean import eog_clean
from .eog_findpeaks import eog_findpeaks


def eog_process(eog_signal, sampling_rate=1000):
    """Process an EOG signal.

    Convenience function that automatically processes an EOG signal.


    Parameters
    ----------
    eog_signal : Union[list, np.array, pd.Series]
        The raw EOG channel, derived from `eog_extract()`.
    sampling_rate : int
        The sampling frequency of `eog_signal` (in Hz, i.e., samples/second). Defaults to 1000.

    Returns
    -------
    signals : DataFrame
        A DataFrame of the same length as the `eog_signal` containing the following columns:
        - *"EOG_Raw"*: the raw signal.
        - *"EOG_Clean"*: the cleaned signal.
        - *"EOG_Blinks"*: the blinks marked as "1" in a list of zeros.
        - *"EOG_Rate"*: eye blinks rate interpolated between blinks.

    info : dict
        A dictionary containing the samples at which the eye blinks occur, accessible with the key "EOG_Blinks".

    See Also
    --------
    eog_extract, eog_clean

    Examples
    --------
    >>> import neurokit2 as nk
    >>> import mne
    >>>
    >>> raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() +
    ...                           '/MEG/sample/sample_audvis_raw.fif', preload=True)
    >>> eog_signal = nk.mne_channel_extract(raw, what='EOG', name='EOG')
    >>> sampling_rate = raw.info['sfreq']
    >>> signals, info = nk.eog_process(eog_signal, sampling_rate=sampling_rate)

    References
    ----------
    - Agarwal, M., & Sivakumar, R. (2019, September). Blink: A Fully Automated Unsupervised Algorithm for
    Eye-Blink Detection in EEG Signals. In 2019 57th Annual Allerton Conference on Communication, Control,
    and Computing (Allerton) (pp. 1113-1121). IEEE.

    """
    # Make sure MNE is installed
#    try:
#        import mne
#    except ImportError:
#        raise ImportError(
#            "NeuroKit error: signal_filter(): the 'mne' module is required for this method to run. "
#            "Please install it first (`pip install mne`).",
#        )

    # Make sure signal is one array
    if isinstance(eog_signal, pd.DataFrame):
        if len(eog_signal.columns) == 2:
            eog_signal = eog_signal.iloc[:, 0] - eog_signal.iloc[:, 1]
        elif len(eog_signal.columns) > 2:
            raise ValueError(
                "NeuroKit warning: eog_process(): Please make sure your EOG signal contains "
                "at most 2 channels of signals."
            )

    # Clean signal
    eog_cleaned = eog_clean(eog_signal, sampling_rate=sampling_rate)

    # Find peaks
    peaks = eog_findpeaks(eog_cleaned, sampling_rate=sampling_rate)

    info = {"EOG_Blinks": peaks}

    # Mark (potential) blink events
    signal_blinks = _signal_from_indices(peaks, desired_length=len(eog_cleaned))

    # Rate computation
    rate = signal_rate(peaks, sampling_rate=sampling_rate, desired_length=len(eog_cleaned))

    # Prepare output
    signals = pd.DataFrame(
        {"EOG_Raw": eog_signal, "EOG_Clean": eog_cleaned, "EOG_Blinks": signal_blinks, "EOG_Rate": rate})

    return signals, info

import numpy as np
import pandas as pd

from ..signal.signal_timefrequency import signal_timefrequency


def entropy_wiener(signal, sampling_rate=1000, **kwargs):
    """Wiener Entropy (WE, also known as Spectral Flatness)

    The Wiener entropy (also known as Spectral Flatness, or tonality coefficient in sound
    processing) is a measure to quantify how noise-like a signal is and is typically applied to
    characterize an audio spectrum.
    A high spectral flatness (closer to 1.0) indicates that the spectrum has a similar amount of
    power in all spectral bands, and is similar to white noise. A low spectral flatness (approaching
    0 for pure tone) indicates that the spectral power is concentrated in a relatively small number of spectral
    bands.

    It is measured on a logarithmic scale from 0 (white noise: log(1): 0) to minus infinity (complete
    order such as pure tone, log(0): minus infinity).

    TODO: double check implementation (especially part on `signal_timefrequency`)

    See Also
    --------
    entropy_spectral

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    **kwargs : optional
        Other arguments to be passed to ``signal_timefrequency()`` (such as 'window').

    Returns
    -------
    ce : float
         The wiener entropy.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute wiener entropy.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(100, sampling_rate=100, frequency=[3, 10])
      we, info = nk.entropy_wiener(signal, sampling_rate=100)
      we

    References
    ----------
    * Wiener, N. (1954). The Human Use of Human Beings: Cybernetics and Society (Boston). Houghton Mifflin, 1, 50.

    * Dubnov, S. (2004). Generalization of spectral flatness measure for non-gaussian linear processes.
    IEEE Signal Processing Letters, 11(8), 698-701.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Get magnitude spectrogram
    _, _, stft = signal_timefrequency(
        signal, sampling_rate=sampling_rate, method="stft", show=False, **kwargs
    )
    # https://github.com/librosa/librosa/blob/eb603e7a91598d1e72d3cdeada0ade21a33f9c0c/librosa/core/spectrum.py#L42

    power = 2
    amin = 1e-10

    S_thresh = np.maximum(amin, stft ** power)
    gmean = np.exp(np.mean(np.log(S_thresh), axis=-2, keepdims=True))[0][0]
    amean = np.mean(S_thresh, axis=-2, keepdims=True)[0][0]

    # Divide geometric mean of power spectrum by arithmetic mean of power spectrum
    return gmean / amean, {}

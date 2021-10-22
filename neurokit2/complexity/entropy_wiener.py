import numpy as np
import pandas as pd
import scipy.signal


def entropy_wiener(signal, sampling_rate=1000, **kwargs):
    """Wiener Entropy (WE, also known as Spectral Flatness)

    The Wiener entropy (also known as Spectral Flatness, or tonality coefficient in sound processing) is a measure to
    quantify how much noise-like a signal is. A high spectral flatness (closer to 1.0) indicates
    that the spectrum is similar to white noise.

    See Also
    --------
    entropy_spectral

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    **kwargs
        Other arguments to be passed to ``signal_psd()`` (such as 'method').

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)

    References
    ----------
    - Kalauzi, A., Bojić, T., & Rakić, L. (2009). Extracting complexity waveforms from one-dimensional signals. Nonlinear biomedical physics, 3(1), 1-11.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # x = [1,4,1,3,5,1,3,6,7]
    # fs = 100
    signal = np.array([1, 4, 1, 3, 5, 1, 3, 6, 7])
    sampling_rate = 100
    _, _, S = specgram = scipy.signal.spectrogram(
        signal,
        fs=sampling_rate,
        window="hamming",
        nfft=1042,
    )

    # Spectrogram (https://github.com/librosa/librosa/blob/eb603e7a91598d1e72d3cdeada0ade21a33f9c0c/librosa/core/spectrum.py#L2468)
    power = 2
    amin = 1e-10

    S_thresh = np.maximum(amin, S ** power)
    gmean = np.exp(np.mean(np.log(S_thresh), axis=-2, keepdims=True))
    amean = np.mean(S_thresh, axis=-2, keepdims=True)
    wiener_entropy = gmean / amean
    return "Not available yet."

import numpy as np
import pandas as pd


def fractal_nld(signal):
    """Fractal dimension via Normalized Length Density (NLD)

    This method was developped for very short epochs durations. TODO: add more 
    information about that.

    See Also
    --------
    fractal_higushi

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> nld, _ = nk.fractal_nld(signal)

    References
    ----------
    - Kalauzi, A., Bojić, T., & Rakić, L. (2009). Extracting complexity waveforms from one-dimensional signals. Nonlinear biomedical physics, 3(1), 1-11.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Based on https://github.com/tfburns/MATLAB-functions-for-complexity-measures-of-one-dimensional-signals/blob/master/nld.m
    # See also https://www.researchgate.net/publication/26743594_Extracting_complexity_waveforms_from_one-dimensional_signals
    n = len(signal)

    # amplitude normalization (could use integral or window normalization techniques instead)
    signal = (np.array(signal) - np.mean(signal)) / np.std(signal, ddof=1)

    # calculate NLDi
    nld = np.concatenate([[0], np.diff(signal)])

    # Use complete algorithm and sum NLDi's
    nld = (1 / n) * np.nansum(nld)

    # Use the power model for NLD to FD conversion
    # a = 1.9079
    # NLDz = 0.097178
    # k = 0.18383
    # a*((NLD-NLDz).^k)
    fd = 1.9079 * np.power(np.array(nld - 0.097178, dtype=complex), 0.18383)
    return fd.real, {}

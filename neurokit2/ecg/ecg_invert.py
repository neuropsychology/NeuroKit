import matplotlib.pyplot as plt
import numpy as np

from .ecg_clean import ecg_clean


def ecg_invert(ecg_signal, sampling_rate=1000, force=False, show=False):
    """**ECG signal inversion**

    Checks whether an ECG signal is inverted, and if so, corrects for this inversion.
    To automatically detect the inversion, the ECG signal is cleaned, the mean is subtracted,
    and with a rolling window of 2 seconds, the original value corresponding to the maximum
    of the squared signal is taken. If the median of these values is negative, it is
    assumed that the signal is inverted.

    Parameters
    ----------
    ecg_signal : Union[list, np.array, pd.Series]
        The raw ECG channel.
    sampling_rate : int
        The sampling frequency of ``ecg_signal`` (in Hz, i.e., samples/second). Defaults to 1000.
    force : bool
        Whether to force inversion of the signal regardless of whether it is
        detected as inverted. The default is False.
    show : bool
        Shows a plot of the original and inverted signal.

    Returns
    -------
    array
        Vector containing the corrected ECG signal.
    bool
        Whether the inversion was performed.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      ecg = -1 * nk.ecg_simulate(duration=10, sampling_rate=200, heart_rate=70)

      # Invert if necessary
      @savefig p_ecg_invert1.png scale=100%
      ecg_fixed, is_inverted = nk.ecg_invert(ecg, sampling_rate=200, show=True)
      @suppress
      plt.close()

    """
    # Invert in any case (needed to perform the check)
    inverted_ecg = np.array(ecg_signal) * -1 + 2 * np.nanmean(ecg_signal)

    if show is True:
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        ax[0].plot(ecg_signal)
        ax[0].set_title("Original ECG")
        ax[1].plot(inverted_ecg)
        ax[1].set_title("Inverted ECG")

    if force:
        was_inverted = True
    else:
        if _ecg_inverted(ecg_signal, sampling_rate=sampling_rate):
            was_inverted = True
        else:
            inverted_ecg = ecg_signal
            was_inverted = False

    return inverted_ecg, was_inverted


def _ecg_inverted(ecg_signal, sampling_rate=1000, window_time=2.0):
    """Checks whether an ECG signal is inverted."""
    ecg_cleaned = ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    # mean should already be close to zero after filtering but just in case, subtract
    ecg_cleaned_meanzero = ecg_cleaned - np.nanmean(ecg_cleaned)
    # take the median of the original value of the maximum of the squared signal
    # over a window where we would expect at least one heartbeat
    med_max_squared = np.nanmedian(
        _roll_orig_max_squared(ecg_cleaned_meanzero, window=int(window_time * sampling_rate))
    )
    # if median is negative, assume inverted
    return med_max_squared < 0


def _roll_orig_max_squared(x, window=2000):
    """With a rolling window, takes the original value corresponding to the maximum of the squared signal."""
    x_rolled = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)
    # https://stackoverflow.com/questions/61703879/in-numpy-how-to-select-elements-based-on-the-maximum-of-their-absolute-values
    shape = np.array(x_rolled.shape)
    shape[-1] = -1
    return np.take_along_axis(x_rolled, np.square(x_rolled).argmax(-1).reshape(shape), axis=-1)

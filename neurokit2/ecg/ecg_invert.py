import numpy as np

from .ecg_clean import ecg_clean


def ecg_invert(ecg_signal, sampling_rate=1000, check_inverted=True):
    """**ECG signal inversion**

    Checks whether an ECG signal is inverted, and if so, corrects for this inversion.

    Parameters
    ----------
    ecg_signal : Union[list, np.array, pd.Series]
        The raw ECG channel.
    sampling_rate : int
        The sampling frequency of ``ecg_signal`` (in Hz, i.e., samples/second). Defaults to 1000.
    check_inverted : bool, optional
        Whether to check whether the signal is inverted before inverting it. The default is True.
        If False, always returns the inverted input signal regardless of whether the input was inverted.
    Returns
    -------
    array
        Vector containing the corrected ECG signal.
    Examples
    --------
    **Example 1**: With an inverted ECG signal
    .. ipython:: python
      import neurokit2 as nk
      import matplotlib.pyplot as plt
      plt.rc('font', size=8)
      # Download data
      data = nk.data("bio_resting_5min_100hz")
      # Invert ECG signal
      ecg_inverted = data["ECG"] * -1 + 2 * np.nanmean(data["ECG"])
      # Fix inversion
      ecg_fixed = nk.ecg_invert(ecg_inverted, sampling_rate=100)

      # Plot inverted ECG and fixed ECG
      @savefig p_ecg_inverted1.png scale=100%
      fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
      ax[0].plot(ecg_inverted[:sampling_rate*5])
      ax[0].set_title("Inverted ECG")
      ax[1].plot(ecg_fixed[:sampling_rate*5])
      ax[1].set_title("Fixed ECG")
      fig.show()
    """
    inverted_ecg_signal = np.array(ecg_signal) * -1 + 2 * np.nanmean(ecg_signal)
    if check_inverted:
        if _ecg_inverted(ecg_signal, sampling_rate=sampling_rate):
            return inverted_ecg_signal
        else:
            return ecg_signal
    else:
        return inverted_ecg_signal


def _ecg_inverted(ecg_signal, sampling_rate=1000):
    """Checks whether an ECG signal is inverted."""
    ecg_cleaned = ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    med_max = np.nanmedian(_roll_func(ecg_cleaned, window=1 * sampling_rate, func=_orig_max_squared))
    return med_max < np.nanmean(ecg_cleaned)


def _roll_func(x, window, func, func_args={}):
    """Applies a function with a rolling window."""
    roll_x = np.array([func(x[i : i + window], **func_args) for i in range(len(x) - window)])
    return roll_x


def _orig_max_squared(x):
    """Returns the original value corresponding to the maximum of the squared signal."""
    return x[np.argmax(np.square(x))]

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .signal_psd import signal_psd


def signal_power(
    signal,
    frequency_band,
    sampling_rate=1000,
    continuous=False,
    show=False,
    normalize=True,
    **kwargs,
):
    """**Compute the power of a signal in a given frequency band**

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    frequency_band :tuple or list
        Tuple or list of tuples indicating the range of frequencies to compute the power in.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    continuous : bool
        Compute instant frequency, or continuous power.
    show : bool
        If ``True``, will return a Poincaré plot. Defaults to ``False``.
    normalize : bool
        Normalization of power by maximum PSD value. Default to ``True``.
        Normalization allows comparison between different PSD methods.
    **kwargs
        Keyword arguments to be passed to :func:`.signal_psd`.

    See Also
    --------
    signal_filter, signal_psd

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Power Spectrum values and a plot if
        ``show`` is ``True``.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      import numpy as np

      # Instant power
      signal = nk.signal_simulate(duration=60, frequency=[10, 15, 20],
                                  amplitude = [1, 2, 3], noise = 2)

      @savefig p_signal_power1.png scale=100%
      power_plot = nk.signal_power(signal, frequency_band=[(8, 12), (18, 22)], method="welch", show=True)
      @suppress
      plt.close()

    ..ipython:: python

      # Continuous (simulated signal)
      signal = np.concatenate((nk.ecg_simulate(duration=30, heart_rate=75), nk.ecg_simulate(duration=30, heart_rate=85)))
      power = nk.signal_power(signal, frequency_band=[(72/60, 78/60), (82/60, 88/60)], continuous=True)
      processed, _ = nk.ecg_process(signal)
      power["ECG_Rate"] = processed["ECG_Rate"]

      @savefig p_signal_power2.png scale=100%
      nk.signal_plot(power, standardize=True)
      @suppress
      plt.close()

    .. ipython:: python

      # Continuous (real signal)
      signal = nk.data("bio_eventrelated_100hz")["ECG"]
      power = nk.signal_power(signal, sampling_rate=100, frequency_band=[(0.12, 0.15), (0.15, 0.4)], continuous=True)
      processed, _ = nk.ecg_process(signal, sampling_rate=100)
      power["ECG_Rate"] = processed["ECG_Rate"]

      @savefig p_signal_power3.png scale=100%
      nk.signal_plot(power, standardize=True)
      @suppress
      plt.close()

    """

    if continuous is False:
        out = _signal_power_instant(
            signal,
            frequency_band,
            sampling_rate=sampling_rate,
            show=show,
            normalize=normalize,
            **kwargs,
        )
    else:
        out = _signal_power_continuous(
            signal, frequency_band, sampling_rate=sampling_rate
        )

    out = pd.DataFrame.from_dict(out, orient="index").T

    return out


# =============================================================================
# Instant
# =============================================================================


def _signal_power_instant(
    signal,
    frequency_band,
    sampling_rate=1000,
    show=False,
    normalize=True,
    order_criteria="KIC",
    **kwargs,
):
    # Sanitize frequency band
    if isinstance(frequency_band[0], (int, float)):
        frequency_band = [frequency_band]  # put in list to iterate on

    #  Get min-max frequency
    min_freq = min([band[0] for band in frequency_band])
    max_freq = max([band[1] for band in frequency_band])

    # Get PSD
    psd = signal_psd(
        signal,
        sampling_rate=sampling_rate,
        show=False,
        normalize=normalize,
        order_criteria=order_criteria,
        **kwargs,
    )

    psd = psd[(psd["Frequency"] >= min_freq) & (psd["Frequency"] <= max_freq)]

    out = {}
    for band in frequency_band:
        power = _signal_power_instant_compute(psd, band)
        out[f"Hz_{band[0]}_{band[1]}"] = power

    if show:
        _signal_power_instant_plot(psd, out, frequency_band)
    return out


def _signal_power_instant_compute(psd, band):
    """Also used in other instances"""
    where = (psd["Frequency"] >= band[0]) & (psd["Frequency"] < band[1])
    power = np.trapz(y=psd["Power"][where], x=psd["Frequency"][where])
    return np.nan if power == 0.0 else power


def _signal_power_instant_plot(psd, out, frequency_band, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    # Sanitize signal
    if isinstance(frequency_band[0], int):
        if len(frequency_band) > 2:
            print(
                "NeuroKit error: signal_power(): The `frequency_band` argument must be a list of tuples"
                " or a tuple of 2 integers"
            )
        else:
            frequency_band = [tuple(i for i in frequency_band)]

    freq = np.array(psd["Frequency"])
    power = np.array(psd["Power"])

    # Get indexes for different frequency band
    frequency_band_index = []
    for band in frequency_band:
        indexes = np.logical_and(
            psd["Frequency"] >= band[0], psd["Frequency"] < band[1]
        )  # pylint: disable=E1111
        frequency_band_index.append(np.array(indexes))

    labels = list(out.keys())
    # Reformat labels if of the pattern "Hz_X_Y"
    if len(labels[0].split("_")) == 3:
        labels = [i.split("_") for i in labels]
        labels = [f"{i[1]}-{i[2]} Hz" for i in labels]

    # Get cmap
    cmap = plt.get_cmap("Set1")
    colors = cmap.colors
    colors = (
        colors[3],
        colors[1],
        colors[2],
        colors[4],
        colors[0],
        colors[5],
        colors[6],
        colors[7],
        colors[8],
    )  # manually rearrange colors
    colors = colors[0 : len(frequency_band_index)]

    # Plot
    ax.set_title("Power Spectral Density (PSD) for Frequency Domains")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Spectrum (ms2/Hz)")

    ax.fill_between(freq, 0, power, color="lightgrey")

    for band_index, label, i in zip(frequency_band_index, labels, colors):
        ax.fill_between(freq[band_index], 0, power[band_index], label=label, color=i)
        ax.legend(prop={"size": 10}, loc="best")

    return fig


# =============================================================================
# Continuous
# =============================================================================


def _signal_power_continuous(signal, frequency_band, sampling_rate=1000):

    out = {}
    if isinstance(frequency_band[0], (list, tuple)):
        for band in frequency_band:
            out.update(_signal_power_continuous_get(signal, band, sampling_rate))
    else:
        out.update(_signal_power_continuous_get(signal, frequency_band, sampling_rate))
    return out


def _signal_power_continuous_get(
    signal, frequency_band, sampling_rate=1000, precision=20
):

    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "NeuroKit error: signal_power(): the 'mne'",
            "module is required. ",
            "Please install it first (`pip install mne`).",
        ) from e  # explicitly raise error from ImportError exception

    out = mne.time_frequency.tfr_array_morlet(
        [[signal]],
        sfreq=sampling_rate,
        freqs=np.linspace(frequency_band[0], frequency_band[1], precision),
        zero_mean=False,
        output="power",
    )
    power = np.mean(out[0][0], axis=0)

    out = {}
    out[f"{frequency_band[0]:.2f}-{frequency_band[1]:.2f}Hz"] = (
        power  # use literal string format
    )
    return out

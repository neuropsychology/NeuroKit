# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np
import pandas as pd
import scipy

from ..misc import NeuroKitWarning
from ..signal import signal_filter, signal_resample, signal_timefrequency
from ..signal.signal_power import _signal_power_instant_compute
from ..signal.signal_psd import _signal_psd_welch
from ..stats import standardize


def eda_sympathetic(
    eda_signal, sampling_rate=1000, frequency_band=[0.045, 0.25], method="posada", show=False
):
    """**Sympathetic Nervous System Index from Electrodermal activity (EDA)**

    Derived from Posada-Quintero et al. (2016), who argue that dynamics of the sympathetic component
    of EDA signal is represented in the frequency band of 0.045-0.25Hz. Note that the Posada method
    requires a signal of a least 64 seconds.

    Parameters
    ----------
    eda_signal : Union[list, np.array, pd.Series]
        The EDA signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    frequency_band : list
        List indicating the frequency range to compute the the power spectral density in.
        Defaults to [0.045, 0.25].
    method : str
        Can be one of ``"ghiasi"`` or ``"posada"``.
    show : bool
        If True, will return a plot of the power spectrum of the EDA signal within the specified
        frequency band.

    See Also
    --------
    .signal_filter, .signal_power, .signal_psd

    Returns
    -------
    dict
        A dictionary containing the EDA sympathetic indexes, accessible by keys
        ``"EDA_Sympathetic"`` and ``"EDA_SympatheticN"`` (normalized, obtained by dividing EDA_Symp
        by total power).

        .. codebookadd::
            EDA_Sympathetic|Derived from Posada-Quintero et al. (2016), who argue that dynamics of \
                the sympathetic component of EDA signal is represented in the frequency band of 0.045-0.25Hz.
            EDA_SympatheticN|normalized version of "EDA_Sympathetic" obtained by dividing \
                EDA_Sympathetic by total power

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      eda = nk.data('bio_resting_8min_100hz')['EDA']

      @savefig p_eda_sympathetic1.png scale=100%
      nk.eda_sympathetic(eda, sampling_rate=100, method='posada', show=True)
      @suppress
      plt.close()

      results = nk.eda_sympathetic(eda, sampling_rate=100, method='ghiasi')
      results

    References
    ----------
    * Ghiasi, S., Grecol, A., Nardelli, M., Catrambonel, V., Barbieri, R., Scilingo, E., & Valenza,
      G. (2018). A New Sympathovagal Balance Index from Electrodermal Activity and Instantaneous
      Vagal Dynamics: A Preliminary Cold Pressor Study. 2018 40th Annual International Conference
      of the IEEE Engineering in Medicine and Biology Society (EMBC). doi:10.1109/embc.2018.8512932
    * Posada-Quintero, H. F., Florian, J. P., Orjuela-Cañón, A. D., Aljama-Corrales, T.,
      Charleston-Villalobos, S., & Chon, K. H. (2016). Power spectral density analysis of
      electrodermal activity for sympathetic function assessment. Annals of biomedical engineering,
      44(10), 3124-3135.

    """

    out = {}

    if method.lower() in ["ghiasi", "ghiasi2018"]:
        out = _eda_sympathetic_ghiasi(
            eda_signal, sampling_rate=sampling_rate, frequency_band=frequency_band, show=show
        )
    elif method.lower() in ["posada", "posada-quintero", "quintero", "posada2016"]:
        out = _eda_sympathetic_posada(
            eda_signal, sampling_rate=sampling_rate, frequency_band=frequency_band, show=show
        )
    else:
        raise ValueError(
            "NeuroKit error: eda_sympathetic(): 'method' should be " "one of 'ghiasi', 'posada'."
        )

    return out


# =============================================================================
# Methods
# =============================================================================


def _eda_sympathetic_posada(
    eda_signal, frequency_band=[0.045, 0.25], sampling_rate=1000, show=True, out={}
):

    # This method assumes signal longer than 64 s
    if len(eda_signal) <= sampling_rate * 64:
        warn(
            "The 'posada2016' method requires a signal of length > 60 s. Try with"
            + " `method='ghiasi2018'`. Returning NaN values for now.",
            category=NeuroKitWarning,
        )
        return {"EDA_Sympathetic": np.nan, "EDA_SympatheticN": np.nan}

    # Resample the eda signal before calculate the synpathetic index based on Posada (2016)
    eda_signal_400hz = signal_resample(
        eda_signal, sampling_rate=sampling_rate, desired_sampling_rate=400
    )

    # 8-th order Chebyshev Type I low-pass filter
    sos = scipy.signal.cheby1(8, 1, 0.8, "lowpass", fs=400, output="sos")
    eda_signal_filtered = scipy.signal.sosfilt(sos, eda_signal_400hz)

    # First step of downsampling
    downsampled_1 = scipy.signal.decimate(eda_signal_filtered, q=10, n=8)  # Keep every 10th sample
    downsampled_2 = scipy.signal.decimate(downsampled_1, q=20, n=8)  # Keep every 20th sample

    # High pass filter
    eda_filtered = signal_filter(
        downsampled_2, sampling_rate=2, lowcut=0.01, highcut=None, method="butterworth", order=8
    )

    nperseg = 128
    overlap = nperseg // 2  # 50 % data overlap

    # Compute psd
    frequency, power = _signal_psd_welch(
        eda_filtered, sampling_rate=2, nperseg=nperseg, window_type="blackman", noverlap=overlap
    )
    psd = pd.DataFrame({"Frequency": frequency, "Power": power})

    # Get sympathetic nervous system indexes
    eda_symp = _signal_power_instant_compute(psd, (frequency_band[0], frequency_band[1]))

    # Compute normalized psd
    psd["Power"] /= np.max(psd["Power"])
    eda_symp_normalized = _signal_power_instant_compute(psd, (frequency_band[0], frequency_band[1]))

    psd_plot = psd.loc[
        np.logical_and(psd["Frequency"] >= frequency_band[0], psd["Frequency"] <= frequency_band[1])
    ]

    if show is True:
        ax = psd_plot.plot(x="Frequency", y="Power", title="EDA Power Spectral Density (us^2/Hz)")
        ax.set(xlabel="Frequency (Hz)", ylabel="Spectrum")

    out = {"EDA_Sympathetic": eda_symp, "EDA_SympatheticN": eda_symp_normalized}

    return out


def _eda_sympathetic_ghiasi(
    eda_signal, sampling_rate=1000, frequency_band=[0.045, 0.25], show=True, out={}
):
    min_frequency = frequency_band[0]
    max_frequency = frequency_band[1]

    # Downsample, normalize, filter
    desired_sampling_rate = 50
    downsampled = signal_resample(
        eda_signal, sampling_rate=sampling_rate, desired_sampling_rate=desired_sampling_rate
    )
    normalized = standardize(downsampled)
    filtered = signal_filter(
        normalized,
        sampling_rate=desired_sampling_rate,
        lowcut=0.01,
        highcut=0.5,
        method="butterworth",
    )

    # Divide the signal into segments and obtain the timefrequency representation
    overlap = 59 * 50  # overlap of 59s in samples

    # TODO: the plot should be improved for this specific case
    _, _, bins = signal_timefrequency(
        filtered,
        sampling_rate=desired_sampling_rate,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        method="stft",
        window=60,
        window_type="blackman",
        overlap=overlap,
        show=show,
    )

    eda_symp = np.mean(bins)
    eda_symp_normalized = eda_symp / np.max(bins)

    out = {"EDA_Sympathetic": eda_symp, "EDA_SympatheticN": eda_symp_normalized}

    return out

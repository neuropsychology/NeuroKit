import matplotlib.pyplot as plt
import numpy as np

from ..ecg.ecg_peaks import _ecg_peaks_plot_artefacts
from ..signal import signal_fixpeaks, signal_formatpeaks, signal_tidypeaksonsets
from ..stats import rescale
from .ppg_findpeaks import ppg_findpeaks


def ppg_peaks(
    ppg_cleaned,
    sampling_rate=1000,
    method="elgendi",
    correct_artifacts=False,
    show=False,
    **kwargs
):
    """**Find systolic peaks in a photoplethysmogram (PPG) signal**

    Find the peaks in an PPG signal using the specified method. You can pass an unfiltered PPG
    signals as input, but typically a filtered PPG (cleaned using ``ppg_clean()``) will result in
    better results.

    .. note::

      Please help us improve the methods' documentation and features.


    Parameters
    ----------
    ppg_cleaned : Union[list, np.array, pd.Series]
        The cleaned PPG channel as returned by ``ppg_clean()``.
    sampling_rate : int
        The sampling frequency of ``ppg_cleaned`` (in Hz, i.e., samples/second). Defaults to 1000.
    method : str
        The processing pipeline to apply. Can be one of ``"elgendi"``, ``"bishop"``, ``"charlton"``.
        The default is ``"elgendi"``.
    correct_artifacts : bool
        Whether or not to identify and fix artifacts, using the method by
        Lipponen & Tarvainen (2019).
    show : bool
        If ``True``, will show a plot of the signal with peaks. Defaults to ``False``.
    **kwargs
        Additional keyword arguments, usually specific for each method.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as the input signal in which occurrences of R-peaks marked as
        ``1`` in a list of zeros with the same length as ``ppg_cleaned``. Accessible with the keys
        ``"PPG_Peaks"``.
    info : dict
        A dictionary containing additional information, in this case the samples at which R-peaks
        occur, accessible with the key ``"PPG_Peaks"``, as well as the signals' sampling rate,
        accessible with the key ``"sampling_rate"``.

    See Also
    --------
    ppg_clean, ppg_fixpeaks, .signal_fixpeaks

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      import numpy as np

      ppg = nk.ppg_simulate(heart_rate=75, duration=20, sampling_rate=50)
      ppg[400:600] = ppg[400:600] + np.random.normal(0, 1.25, 200)

      # Default method (Elgendi et al., 2013)
      @savefig p_ppg_peaks1.png scale=100%
      peaks, info = nk.ppg_peaks(ppg, sampling_rate=100, method="elgendi", show=True)
      @suppress
      plt.close()
      info["PPG_Peaks"]

      # Method by Bishop et al., (2018)
      @savefig p_ppg_peaks2.png scale=100%
      peaks, info = nk.ppg_peaks(ppg, sampling_rate=100, method="bishop", show=True)
      @suppress
      plt.close()

      # Correct artifacts
      @savefig p_ppg_peaks3.png scale=100%
      peaks, info = nk.ppg_peaks(ppg, sampling_rate=100, correct_artifacts=True, show=True)
      @suppress
      plt.close()

    References
    ----------
    * Elgendi, M., Norton, I., Brearley, M., Abbott, D., & Schuurmans, D. (2013). Systolic peak
      detection in acceleration photoplethysmograms measured from emergency responders in tropical
      conditions. PloS one, 8(10), e76585.
    * Bishop, S. M., & Ercole, A. (2018). Multi-scale peak and trough detection optimised for
      periodic and quasi-periodic neuroscience data. In Intracranial Pressure & Neuromonitoring XVI
      (pp. 189-195). Springer International Publishing.
    * Charlton, P. H. et al. (2025). The MSPTDfast photoplethysmography beat detection algorithm:
      design, benchmarking, and open-source distribution. Physiological Measurement, 46, 035002.

    """
    # Store info
    info = {"method_peaks": method.lower(), "method_fixpeaks": "None"}

    info.update(
        ppg_findpeaks(
            ppg_cleaned,
            sampling_rate=sampling_rate,
            method=method,
            show=False,
            **kwargs
        )
    )

    # Peak (and onset) correction
    # - tidy up peaks and onsets
    if info['method_fixpeaks'].lower() == "charlton2022":  # this is the default settings when using MSPTDfastv1 or MSPTDfastv2
        info["PPG_Peaks_Unfixed"] = info["PPG_Peaks"].copy()
        info["PPG_Onsets_Unfixed"] = info["PPG_Onsets"].copy()

        fixpeaks, info["PPG_Peaks"], info["PPG_Onsets"] = signal_tidypeaksonsets(
            ppg_cleaned, info["PPG_Peaks"], info["PPG_Onsets"], method="Charlton2022"
        )

        # Add prefix and merge
        fixpeaks = {"PPG_fixpeaks_" + str(key): val for key, val in fixpeaks.items()}
        info.update(fixpeaks)


    # - perform peak correction
    if correct_artifacts:
        info["PPG_Peaks_Uncorrected"] = info["PPG_Peaks"].copy()

        fixpeaks, info["PPG_Peaks"] = signal_fixpeaks(
            info["PPG_Peaks"], sampling_rate=sampling_rate, method="Kubios"
        )

        # Add prefix and merge
        fixpeaks = {"PPG_fixpeaks_" + str(key): val for key, val in fixpeaks.items()}
        info.update(fixpeaks)
    
    # Format output
    signals = signal_formatpeaks(
        dict(PPG_Peaks=info["PPG_Peaks"]),
        desired_length=len(ppg_cleaned),
        peak_indices=info["PPG_Peaks"],
    )
    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    if show is True:
        _ppg_peaks_plot(ppg_cleaned, info, sampling_rate)

    return signals, info


# =============================================================================
# Internals
# =============================================================================
def _ppg_peaks_plot(
    ppg_cleaned,
    info=None,
    sampling_rate=1000,
    raw=None,
    quality=None,
    ax=None,
):
    x_axis = np.linspace(0, len(ppg_cleaned) / sampling_rate, len(ppg_cleaned))

    # Prepare plot
    if ax is None:
        _, ax = plt.subplots()

    ax.set_xlabel("Time (seconds)")
    ax.set_title("PPG signal and peaks")

    # Quality Area -------------------------------------------------------------
    if quality is not None:
        quality = rescale(
            quality,
            to=[
                np.min([np.min(raw), np.min(ppg_cleaned)]),
                np.max([np.max(raw), np.max(ppg_cleaned)]),
            ],
        )
        minimum_line = np.full(len(x_axis), quality.min())

        # Plot quality area first
        ax.fill_between(
            x_axis,
            minimum_line,
            quality,
            alpha=0.12,
            zorder=0,
            interpolate=True,
            facecolor="#4CAF50",
            label="Signal quality",
        )

    # Raw Signal ---------------------------------------------------------------
    if raw is not None:
        ax.plot(x_axis, raw, color="#B0BEC5", label="Raw signal", zorder=1)
        label_clean = "Cleaned signal"
    else:
        label_clean = "Signal"

    # Peaks -------------------------------------------------------------------
    ax.scatter(
        x_axis[info["PPG_Peaks"]],
        ppg_cleaned[info["PPG_Peaks"]],
        color="#FFC107",
        label="Systolic peaks",
        zorder=2,
    )

    # Artifacts ---------------------------------------------------------------
    _ecg_peaks_plot_artefacts(
        x_axis,
        ppg_cleaned,
        info,
        info["PPG_Peaks"],
        ax,
    )

    # Clean Signal ------------------------------------------------------------
    ax.plot(
        x_axis,
        ppg_cleaned,
        color="#E91E63",
        label=label_clean,
        zorder=3,
        linewidth=1,
    )

    # Optimize legend
    ax.legend(loc="upper right")

    return ax

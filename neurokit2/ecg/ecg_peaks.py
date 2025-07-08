import matplotlib.pyplot as plt
import numpy as np

from ..signal import signal_fixpeaks, signal_formatpeaks
from ..stats import rescale
from .ecg_findpeaks import ecg_findpeaks


def ecg_peaks(ecg_cleaned, sampling_rate=1000, method="neurokit", correct_artifacts=False, show=False, **kwargs):
    """**Find R-peaks in an ECG signal**

    Find R-peaks in an ECG signal using the specified method. You can pass an unfiltered ECG
    signals as input, but typically a filtered ECG (cleaned using ``ecg_clean()``) will result in
    better results.

    Different algorithms for peak-detection include:

    * **neurokit** (default): QRS complexes are detected based on the steepness of the absolute
      gradient of the ECG signal. Subsequently, R-peaks are detected as local maxima in
      the QRS complexes. The method is unpublished, but see: (i) https://github.com/neuropsychology/NeuroKit/issues/476
      for discussion of this algorithm; and (ii) https://doi.org/10.21105/joss.02621 for the original validation of
      this algorithm.
    * **pantompkins1985**: Algorithm by Pan & Tompkins (1985).
    * **hamilton2002**: Algorithm by Hamilton (2002).
    * **zong2003**: Algorithm by Zong et al. (2003).
    * **martinez2004**: Algorithm by Martinez et al (2004).
    * **christov2004**: Algorithm by Christov (2004).
    * **gamboa2008**: Algorithm by Gamboa (2008).
    * **elgendi2010**: Algorithm by Elgendi et al. (2010).
    * **engzeemod2012**: Original algorithm by Engelse & Zeelenberg (1979) modified by Lourenço et
      al. (2012).
    * **manikandan2012**: Algorithm by Manikandan & Soman (2012) based on the Shannon energy
      envelope (SEE).
    * **khamis2016**: UNSW Algorithm by Khamis et al. (2016), designed for both clinical ECGs and poorer quality
      telehealth ECGs.
    * **kalidas2017**: Algorithm by Kalidas et al. (2017).
    * **nabian2018**: Algorithm by Nabian et al. (2018) based on the Pan-Tompkins algorithm.
    * **rodrigues2021**: Adaptation of the work by Sadhukhan & Mitra (2012) and Gutiérrez-Rivas et
      al. (2015) by Rodrigues et al. (2021).
    * **emrich2023**: FastNVG Algorithm by Emrich et al. (2023) based on the visibility graph detector of Koka et al. (2022).
      Provides fast and sample-accurate R-peak detection. The algorithm transforms the ecg into a graph representation
      and extracts exact R-peak positions using graph metrics.
    * **promac**: ProMAC combines the result of several R-peak detectors in a probabilistic way.
      For a given peak detector, the binary signal representing the peak locations is convolved
      with a Gaussian distribution, resulting in a probabilistic representation of each peak
      location. This procedure is repeated for all selected methods and the resulting
      signals are accumulated. Finally, a threshold is used to accept or reject the peak locations.
      See this discussion for more information on the origins of the method:
      https://github.com/neuropsychology/NeuroKit/issues/222


    .. note::

      Please help us improve the methods' documentation by adding a small description.


    Parameters
    ----------
    ecg_cleaned : Union[list, np.array, pd.Series]
        The cleaned ECG channel as returned by ``ecg_clean()``.
    sampling_rate : int
        The sampling frequency of ``ecg_cleaned`` (in Hz, i.e., samples/second). Defaults to 1000.
    method : string
        The algorithm to be used for R-peak detection.
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
        ``1`` in a list of zeros with the same length as ``ecg_cleaned``. Accessible with the keys
        ``"ECG_R_Peaks"``.
    info : dict
        A dictionary containing additional information, in this case the samples at which R-peaks
        occur, accessible with the key ``"ECG_R_Peaks"``, as well as the signals' sampling rate,
        accessible with the key ``"sampling_rate"``.

    See Also
    --------
    ecg_clean, ecg_findpeaks, .signal_fixpeaks

    Examples
    --------
    * **Example 1**: Find R-peaks using the default method (``"neurokit"``).

    .. ipython:: python

      import neurokit2 as nk
      import numpy as np

      ecg = nk.ecg_simulate(duration=10, sampling_rate=250)
      ecg[600:950] = ecg[600:950] + np.random.normal(0, 0.6, 350)

      @savefig p_ecg_peaks1.png scale=100%
      signals, info = nk.ecg_peaks(ecg, sampling_rate=250, correct_artifacts=True, show=True)
      @suppress
      plt.close()

    * **Example 2**: Compare different methods

    .. ipython:: python

      # neurokit (default)
      cleaned = nk.ecg_clean(ecg, sampling_rate=250, method="neurokit")
      _, neurokit = nk.ecg_peaks(cleaned, sampling_rate=250, method="neurokit")

      # pantompkins1985
      cleaned = nk.ecg_clean(ecg, sampling_rate=250, method="pantompkins1985")
      _, pantompkins1985 = nk.ecg_peaks(cleaned, sampling_rate=250, method="pantompkins1985")

      # hamilton2002
      cleaned = nk.ecg_clean(ecg, sampling_rate=250, method="hamilton2002")
      _, hamilton2002 = nk.ecg_peaks(cleaned, sampling_rate=250, method="hamilton2002")

      # zong2003
      _, zong2003 = nk.ecg_peaks(ecg, sampling_rate=250, method="zong2003")

      # martinez2004
      _, martinez2004 = nk.ecg_peaks(ecg, sampling_rate=250, method="martinez2004")

      # christov2004
      _, christov2004 = nk.ecg_peaks(cleaned, sampling_rate=250, method="christov2004")

      # gamboa2008
      cleaned = nk.ecg_clean(ecg, sampling_rate=250, method="gamboa2008")
      _, gamboa2008 = nk.ecg_peaks(cleaned, sampling_rate=250, method="gamboa2008")

      # elgendi2010
      cleaned = nk.ecg_clean(ecg, sampling_rate=250, method="elgendi2010")
      _, elgendi2010 = nk.ecg_peaks(cleaned, sampling_rate=250, method="elgendi2010")

      # engzeemod2012
      cleaned = nk.ecg_clean(ecg, sampling_rate=250, method="engzeemod2012")
      _, engzeemod2012 = nk.ecg_peaks(cleaned, sampling_rate=250, method="engzeemod2012")

      # Manikandan (2012)
      _, manikandan2012 = nk.ecg_peaks(ecg, sampling_rate=250, method="manikandan2012")

      # Khamis (2016)
      _, khamis2016 = nk.ecg_peaks(ecg, sampling_rate=250, method="khamis2016")

      # kalidas2017
      cleaned = nk.ecg_clean(ecg, sampling_rate=250, method="kalidas2017")
      _, kalidas2017 = nk.ecg_peaks(cleaned, sampling_rate=250, method="kalidas2017")

      # nabian2018
      _, nabian2018 = nk.ecg_peaks(ecg, sampling_rate=250, method="nabian2018")

      # rodrigues2021
      _, rodrigues2021 = nk.ecg_peaks(ecg, sampling_rate=250, method="rodrigues2021")

      # emrich2023
      cleaned = nk.ecg_clean(ecg, sampling_rate=250, method="emrich2023")
      _, emrich2023 = nk.ecg_peaks(cleaned, sampling_rate=250, method="emrich2023")

      # Collect all R-peak lists by iterating through the result dicts
      rpeaks = [
          i["ECG_R_Peaks"]
          for i in [
              neurokit,
              pantompkins1985,
              nabian2018,
              hamilton2002,
              martinez2004,
              christov2004,
              gamboa2008,
              elgendi2010,
              engzeemod2012,
              khamis2016,
              kalidas2017,
              rodrigues2021,
              emrich2023
          ]
      ]
      # Visualize results
      @savefig p_ecg_peaks2.png scale=100%
      nk.events_plot(rpeaks, ecg)
      @suppress
      plt.close()

    * **Example 3**: Method-agreement procedure ('promac')

    .. ipython:: python

      ecg = nk.ecg_simulate(duration=10, sampling_rate=500)
      ecg = nk.signal_distort(ecg,
                              sampling_rate=500,
                              noise_amplitude=0.05, noise_frequency=[25, 50],
                              artifacts_amplitude=0.05, artifacts_frequency=50)
      @savefig p_ecg_peaks3.png scale=100%
      info = nk.ecg_findpeaks(ecg, sampling_rate=250, method="promac", show=True)
      @suppress
      plt.close()

    References
    ----------
    * Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE transactions
      on biomedical engineering, (3), 230-236.
    * Hamilton, P. (2002). Open source ECG analysis. In Computers in cardiology (pp. 101-104).
      IEEE.
    * Zong, W., Heldt, T., Moody, G. B., & Mark, R. G. (2003). An open-source algorithm to
      detect onset of arterial blood pressure pulses. In Computers in Cardiology, 2003 (pp.
      259-262). IEEE.
    * Zong, W., Moody, G. B., & Jiang, D. (2003, September). A robust open-source algorithm to
      detect onset and duration of QRS complexes. In Computers in Cardiology, 2003 (pp.
      737-740). IEEE.
    * Martinez, J. P., Almeida, R., Olmos, S., Rocha, A. P., & Laguna, P. (2004) A wavelet-based
      ECG delineator: evaluation on standard databases. IEEE Trans Biomed Eng, 51(4), 570–581.
    * Christov, I. I. (2004). Real time electrocardiogram QRS detection using combined adaptive
      threshold. Biomedical engineering online, 3(1), 1-9.
    * Gamboa, H. (2008). Multi-modal behavioral biometrics based on HCI and electrophysiology
      (Doctoral dissertation, Universidade Técnica de Lisboa).
    * Elgendi, M., Jonkman, M., & De Boer, F. (2010). Frequency Bands Effects on QRS Detection.
      Biosignals, Proceedings of the Third International Conference on Bio-inspired Systems and
      Signal Processing, 428-431.
    * Engelse, W. A., & Zeelenberg, C. (1979). A single scan algorithm for QRS-detection and
      feature extraction. Computers in cardiology, 6(1979), 37-42.
    * Manikandan, M. S., & Soman, K. P. (2012). A novel method for detecting R-peaks in
      electrocardiogram (ECG) signal. Biomedical Signal Processing and Control, 7(2), 118-128.
    * Lourenço, A., Silva, H., Leite, P., Lourenço, R., & Fred, A. L. (2012, February). Real
      Time Electrocardiogram Segmentation for Finger based ECG Biometrics. In Biosignals (pp.
      49-54).
    * Khamis, H., Weiss, R., Xie, Y., Chang, C. W., Lovell, N. H., & Redmond, S. J. (2016).
      QRS detection algorithm for telehealth electrocardiogram recordings.
      IEEE Transactions on Biomedical Engineering, 63(7), 1377–1388.
    * Kalidas, V., & Tamil, L. (2017, October). Real-time QRS detector using stationary wavelet
      transform for automated ECG analysis. In 2017 IEEE 17th International Conference on
      Bioinformatics and Bioengineering (BIBE) (pp. 457-461). IEEE.
    * Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., Ostadabbas, S. (2018).
      An Open-Source Feature Extraction Tool for the Analysis of Peripheral Physiological Data.
      IEEE Journal of Translational Engineering in Health and Medicine, 6, 1-11.
    * Sadhukhan, D., & Mitra, M. (2012). R-peak detection algorithm for ECG using double
      difference and RR interval processing. Procedia Technology, 4, 873-877.
    * Gutiérrez-Rivas, R., García, J. J., Marnane, W. P., & Hernández, A. (2015). Novel
      real-time low-complexity QRS complex detector based on adaptive thresholding. IEEE
      Sensors Journal, 15(10), 6036-6043.
    * Rodrigues, T., Samoutphonh, S., Silva, H., & Fred, A. (2021, January). A Low-Complexity
      R-peak Detection Algorithm with Adaptive Thresholding for Wearable Devices. In 2020 25th
      International Conference on Pattern Recognition (ICPR) (pp. 1-8). IEEE.
    * T. Koka and M. Muma, "Fast and Sample Accurate R-Peak Detection for Noisy ECG Using
      Visibility Graphs," 2022 44th Annual International Conference of the IEEE Engineering in
      Medicine & Biology Society (EMBC), 2022, pp. 121-126.
    * ``promac``
        * **Unpublished.** It runs different methods and derives a probability index using
          convolution. See this discussion for more information on the method:
          https://github.com/neuropsychology/NeuroKit/issues/222
    * Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability
      time series artefact correction using novel beat classification. Journal of medical
      engineering & technology, 43(3), 173-181.
    * Emrich, J., Koka, T., Wirth, S., & Muma, M. (2023), Accelerated Sample-Accurate R-Peak
      Detectors Based on Visibility Graphs. 31st European Signal Processing Conference
      (EUSIPCO), 1090-1094, doi: 10.23919/EUSIPCO58844.2023.10290007

    """
    # Store info
    info = {"method_peaks": method.lower(), "method_fixpeaks": "None"}

    # First peak detection
    info.update(ecg_findpeaks(ecg_cleaned, sampling_rate=sampling_rate, method=info["method_peaks"], **kwargs))

    # Peak correction
    if correct_artifacts:
        info["ECG_R_Peaks_Uncorrected"] = info["ECG_R_Peaks"].copy()

        fixpeaks, info["ECG_R_Peaks"] = signal_fixpeaks(
            info["ECG_R_Peaks"], sampling_rate=sampling_rate, method="Kubios"
        )

        # Add prefix and merge
        fixpeaks = {"ECG_fixpeaks_" + str(key): val for key, val in fixpeaks.items()}
        info.update(fixpeaks)

    # Format output
    signals = signal_formatpeaks(
        dict(ECG_R_Peaks=info["ECG_R_Peaks"]),  # Takes a dict as input
        desired_length=len(ecg_cleaned),
        peak_indices=info["ECG_R_Peaks"],
    )

    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    if show is True:
        _ecg_peaks_plot(ecg_cleaned, info, sampling_rate)

    return signals, info


# =============================================================================
# Internals
# =============================================================================
def _ecg_peaks_plot(
    ecg_cleaned,
    info=None,
    sampling_rate=1000,
    raw=None,
    quality=None,
    phase=None,
    ax=None,
):
    x_axis = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned))

    # Prepare plot
    if ax is None:
        _, ax = plt.subplots()

    ax.set_xlabel("Time (seconds)")
    ax.set_title("ECG signal and peaks")

    # Quality Area -------------------------------------------------------------
    if quality is not None:
        quality = rescale(
            quality,
            to=[
                np.min([np.min(raw), np.min(ecg_cleaned)]),
                np.max([np.max(raw), np.max(ecg_cleaned)]),
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
        x_axis[info["ECG_R_Peaks"]],
        ecg_cleaned[info["ECG_R_Peaks"]],
        color="#FFC107",
        label="R-peaks",
        zorder=2,
    )

    # Artifacts ---------------------------------------------------------------
    _ecg_peaks_plot_artefacts(
        x_axis,
        ecg_cleaned,
        info,
        peaks=info["ECG_R_Peaks"],
        ax=ax,
    )

    # Clean Signal ------------------------------------------------------------
    if phase is not None:
        mask = (phase == 0) | (np.isnan(phase))
        diastole = ecg_cleaned.copy()
        diastole[~mask] = np.nan

        # Create overlap to avoid interuptions in signal
        mask[np.where(np.diff(mask))[0] + 1] = True
        systole = ecg_cleaned.copy()
        systole[mask] = np.nan

        ax.plot(
            x_axis,
            diastole,
            color="#B71C1C",
            label=label_clean,
            zorder=3,
            linewidth=1,
        )
        ax.plot(
            x_axis,
            systole,
            color="#F44336",
            zorder=3,
            linewidth=1,
        )
    else:
        ax.plot(
            x_axis,
            ecg_cleaned,
            color="#F44336",
            label=label_clean,
            zorder=3,
            linewidth=1,
        )

    # Optimize legend
    if raw is not None:
        handles, labels = ax.get_legend_handles_labels()
        order = [2, 0, 1, 3]
        ax.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc="upper right",
        )
    else:
        ax.legend(loc="upper right")

    return ax


def _ecg_peaks_plot_artefacts(
    x_axis,
    signal,
    info,
    peaks,
    ax,
):
    raw = [s for s in info.keys() if str(s).endswith("Peaks_Uncorrected")]
    if len(raw) == 0:
        return "No correction"
    raw = info[raw[0]]
    if len(raw) == 0:
        return "No bad peaks"
    if any([i < len(signal) for i in raw]):
        return "Peak indices longer than signal. Signals might have been cropped. " + "Better skip plotting."

    extra = [i for i in raw if i not in peaks]
    if len(extra) > 0:
        ax.scatter(
            x_axis[extra],
            signal[extra],
            color="#4CAF50",
            label="Peaks removed after correction",
            marker="x",
            zorder=2,
        )

    added = [i for i in peaks if i not in raw]
    if len(added) > 0:
        ax.scatter(
            x_axis[added],
            signal[added],
            color="#FF9800",
            label="Peaks added after correction",
            marker="x",
            zorder=2,
        )
    return ax

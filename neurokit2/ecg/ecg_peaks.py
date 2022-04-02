# - * - coding: utf-8 - * -

from ..signal import signal_fixpeaks, signal_formatpeaks
from .ecg_findpeaks import ecg_findpeaks


def ecg_peaks(
    ecg_cleaned, sampling_rate=1000, method="neurokit", correct_artifacts=False, **kwargs
):
    """**Find R-peaks in an ECG signal**

    Find R-peaks in an ECG signal using the specified method. The method accepts unfiltered ECG
    signals as input, although it is expected that a filtered (cleaned) ECG will result in better
    results.

    Parameters
    ----------
    ecg_cleaned : Union[list, np.array, pd.Series]
        The cleaned ECG channel as returned by `ecg_clean()`.
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second). Defaults to 1000.
    method : string
        The algorithm to be used for R-peak detection. Can be one of ``'neurokit'`` (default),
        ``'pantompkins1985'``, ``'nabian2018'``, ``'gamboa2008'``, ``'zong2003'``,
        ``'hamilton2002'``, ``'christov2004'``, ``'engzeemod2012'``, ``'elgendi2010'``,
        ``'kalidas2017'``, ``'martinez2003'``, ``'rodrigues2021'`` or ``'promac'``.
    correct_artifacts : bool
        Whether or not to first identify and fix artifacts as defined by
        Lipponen & Tarvainen (2019).
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
    ecg_clean, ecg_findpeaks, signal_fixpeaks

    Examples
    --------
    * **Example 1**: Find R-peaks using the default method (``'neurokit'``).

    .. ipython:: python

      import neurokit2 as nk

      ecg = nk.ecg_simulate(duration=10, sampling_rate=1000)
      signals, info = nk.ecg_peaks(ecg, correct_artifacts=True)

      @savefig p_ecg_peaks1.png scale=100%
      nk.events_plot(info["ECG_R_Peaks"], ecg)

    * **Example 2**: Compare different methods

    .. ipython:: python

      # neurokit (default)
      cleaned = nk.ecg_clean(ecg, method="neurokit")
      _, neurokit = nk.ecg_peaks(cleaned, method="neurokit")

      # pantompkins1985
      cleaned = nk.ecg_clean(ecg, method="pantompkins1985")
      _, pantompkins1985 = nk.ecg_peaks(cleaned, method="pantompkins1985")

      # nabian2018
      _, nabian2018 = nk.ecg_peaks(ecg, method="nabian2018")

      # hamilton2002
      cleaned = nk.ecg_clean(ecg, method="hamilton2002")
      _, hamilton2002 = nk.ecg_peaks(cleaned, method="hamilton2002")

      # martinez2003
      _, martinez2003 = nk.ecg_peaks(ecg, method="martinez2003")

      # christov2004
      _, christov2004 = nk.ecg_peaks(cleaned, method="christov2004")

      # gamboa2008
      cleaned = nk.ecg_clean(ecg, method="gamboa2008")
      _, gamboa2008 = nk.ecg_peaks(cleaned, method="gamboa2008")

      # elgendi2010
      cleaned = nk.ecg_clean(ecg, method="elgendi2010")
      _, elgendi2010 = nk.ecg_peaks(cleaned, method="elgendi2010")

      # engzeemod2012
      cleaned = nk.ecg_clean(ecg, method="engzeemod2012")
      _, engzeemod2012 = nk.ecg_peaks(cleaned, method="engzeemod2012")

      # kalidas2017
      cleaned = nk.ecg_clean(ecg, method="kalidas2017")
      _, kalidas2017 = nk.ecg_peaks(cleaned, method="kalidas2017")

      # rodrigues2021
      _, rodrigues2021 = nk.ecg_peaks(ecg, method="rodrigues2021")

      # Collect all R-peak lists by iterating through the result dicts
      rpeaks = [
          i["ECG_R_Peaks"]
          for i in [
              neurokit,
              pantompkins1985,
              nabian2018,
              hamilton2002,
              martinez2003,
              christov2004,
              gamboa2008,
              elgendi2010,
              engzeemod2012,
              kalidas2017,
              rodrigues2021,
          ]
      ]
      # Visualize results
      @savefig p_ecg_peaks2.png scale=100%
      nk.events_plot(rpeaks, ecg)

    * **Example 3**: Method-agreement procedure ('promac')

    .. ipython:: python

      ecg = nk.ecg_simulate(duration=10, sampling_rate=500)
      ecg = nk.signal_distort(ecg,
                              sampling_rate=500,
                              noise_amplitude=0.05, noise_frequency=[25, 50],
                              artifacts_amplitude=0.05, artifacts_frequency=50)
      @savefig p_ecg_peaks3.png scale=100%
      info = nk.ecg_findpeaks(ecg, sampling_rate=1000, method="promac", show=True)

    References
    ----------
    * ``neurokit``
        Unpublished. See this discussion for more information on the method:
        https://github.com/neuropsychology/NeuroKit/issues/476
    * ``pantompkins1985``
        * Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE transactions
          on biomedical engineering, (3), 230-236.
    * ``nabian2018``
        * Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., Ostadabbas, S. (2018).
          An Open-Source Feature Extraction Tool for the Analysis of Peripheral Physiological Data.
          IEEE Journal of Translational Engineering in Health and Medicine, 6, 1-11.
          doi:10.1109/jtehm.2018.2878000
    * ``gamboa2008``
        * Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci and electrophysiology.
          PhD ThesisUniversidade.
    * ``zong2003``
        * Zong, W., Heldt, T., Moody, G. B., & Mark, R. G. (2003). An open-source algorithm to
          detect onset of arterial blood pressure pulses. In Computers in Cardiology, 2003 (pp.
          259-262). IEEE.
    * ``hamilton2002``
        * Hamilton, P. (2002). Open source ECG analysis. In Computers in cardiology (pp. 101-104).
          IEEE.
    * ``christov2004``
        * Ivaylo I. Christov, Real time electrocardiogram QRS detection using combined adaptive
          threshold, BioMedical Engineering OnLine 2004, vol. 3:28, 2004.
    * ``engzeemod2012``
        * Engelse, W. A., & Zeelenberg, C. (1979). A single scan algorithm for QRS-detection and
          feature extraction. Computers in cardiology, 6(1979), 37-42.
        * Lourenço, A., Silva, H., Leite, P., Lourenço, R., & Fred, A. L. (2012, February). Real
          Time Electrocardiogram Segmentation for Finger based ECG Biometrics. In Biosignals (pp.
          49-54).
    * ``elgendi2010``
        * Elgendi, M., Jonkman, M., & De Boer, F. (2010). Frequency Bands Effects on QRS Detection.
          Biosignals, Proceedings of the Third International Conference on Bio-inspired Systems and
          Signal Processing, 428-431.
    * ``kalidas2017``
        * Kalidas, V., & Tamil, L. (2017, October). Real-time QRS detector using stationary wavelet
          transform for automated ECG analysis. In 2017 IEEE 17th International Conference on
          Bioinformatics and Bioengineering (BIBE) (pp. 457-461). IEEE.
    * ``martinez2003``
        **Unknown.** Please help us retrieve the correct source!
    * ``rodrigues2021``
        * Gutiérrez-Rivas, R., García, J. J., Marnane, W. P., & Hernández, A. (2015). Novel
          real-time low-complexity QRS complex detector based on adaptive thresholding. IEEE
          Sensors Journal, 15(10), 6036-6043.
        * Sadhukhan, D., & Mitra, M. (2012). R-peak detection algorithm for ECG using double
          difference and RR interval processing. Procedia Technology, 4, 873-877.
        * Rodrigues, Tiago & Samoutphonh, Sirisack & Plácido da Silva, Hugo & Fred, Ana. (2021).
          A Low-Complexity R-peak Detection Algorithm with Adaptive Thresholding for Wearable
          Devices.
    * ``promac``
        Unpublished. It runs different methods and derives a probability index using convolution.
        See this discussion for more information on the method:
        https://github.com/neuropsychology/NeuroKit/issues/222
    * Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability
      time series artefact correction using novel beat classification. Journal of medical
      engineering & technology, 43(3), 173-181.

    """
    rpeaks = ecg_findpeaks(ecg_cleaned, sampling_rate=sampling_rate, method=method, **kwargs)

    if correct_artifacts:
        _, rpeaks = signal_fixpeaks(
            rpeaks, sampling_rate=sampling_rate, iterative=True, method="Kubios"
        )

        rpeaks = {"ECG_R_Peaks": rpeaks}

    instant_peaks = signal_formatpeaks(rpeaks, desired_length=len(ecg_cleaned), peak_indices=rpeaks)
    signals = instant_peaks
    info = rpeaks
    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    return signals, info

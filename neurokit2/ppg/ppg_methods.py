# -*- coding: utf-8 -*-
import numpy as np

from ..misc.report import get_kwargs
from .ppg_clean import ppg_clean
from .ppg_findpeaks import ppg_findpeaks
from .ppg_quality import ppg_quality


def ppg_methods(
    sampling_rate=1000,
    method="elgendi",
    method_cleaning="default",
    method_peaks="default",
    method_quality="default",
    **kwargs,
):
    """**PPG Preprocessing Methods**

    This function analyzes and specifies the methods used in the preprocessing, and create a
    textual description of the methods used. It is used by :func:`ppg_process()` to dispatch the
    correct methods to each subroutine of the pipeline and :func:`ppg_report()` to create a
    preprocessing report.

    Parameters
    ----------
    sampling_rate : int
        The sampling frequency of the raw PPG signal (in Hz, i.e., samples/second).
    method : str
        The method used for cleaning and peak finding if ``"method_cleaning"``
        and ``"method_peaks"`` are set to ``"default"``. Can be one of ``"elgendi"``.
        Defaults to ``"elgendi"``.
    method_cleaning: str
        The method used to clean the raw PPG signal. If ``"default"``,
        will be set to the value of ``"method"``. Defaults to ``"default"``.
        For more information, see the ``"method"`` argument
        of :func:`.ppg_clean`.
    method_peaks: str
        The method used to find peaks. If ``"default"``,
        will be set to the value of ``"method"``. Defaults to ``"default"``.
        For more information, see the ``"method"`` argument
        of :func:`.ppg_findpeaks`.
    method_quality: str
        The method used to assess PPG signal quality. If ``"default"``,
        will be set to the value of ``"templatematch"``. Defaults to ``"templatematch"``.
        For more information, see the ``"method"`` argument
        of :func:`.ppg_quality`.
    **kwargs
        Other arguments to be passed to :func:`.ppg_clean` and
        :func:`.ppg_findpeaks`.

    Returns
    -------
    report_info : dict
        A dictionary containing the keyword arguments passed to the cleaning
        and peak finding functions, text describing the methods, and the corresponding
        references.

    See Also
    --------
    ppg_process, ppg_clean, ppg_findpeaks, ppg_quality

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      methods = nk.ppg_methods(
          sampling_rate=100, method="elgendi",
          method_cleaning="nabian2018", method_quality="templatematch")
      print(methods["text_cleaning"])
      print(methods["references"][0])

    """
    # Sanitize inputs
    method_cleaning = (
        str(method).lower()
        if method_cleaning == "default"
        else str(method_cleaning).lower()
    )
    method_peaks = (
        str(method).lower()
        if method_peaks == "default"
        else str(method_peaks).lower()
    )
    method_quality = (
        str(method_quality).lower()
    )

    # Create dictionary with all inputs
    report_info = {
        "sampling_rate": sampling_rate,
        "method": method,
        "method_cleaning": method_cleaning,
        "method_peaks": method_peaks,
        "method_quality": method_quality,
        **kwargs,
    }

    # Get arguments to be passed to cleaning, peak finding, and quality assessment functions
    kwargs_cleaning, report_info = get_kwargs(report_info, ppg_clean)
    kwargs_peaks, report_info = get_kwargs(report_info, ppg_findpeaks)
    kwargs_quality, report_info = get_kwargs(report_info, ppg_quality)

    # Save keyword arguments in dictionary
    report_info["kwargs_cleaning"] = kwargs_cleaning
    report_info["kwargs_peaks"] = kwargs_peaks
    report_info["kwargs_quality"] = kwargs_quality

    # Initialize refs list with NeuroKit2 reference
    refs = ["""Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H.,
    Schölzel, C., & Chen, S. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing.
    Behavior Research Methods, 53(4), 1689–1696. https://doi.org/10.3758/s13428-020-01516-y
    """]

    # 1. Cleaning
    # ------------
    report_info["text_cleaning"] = f"The raw signal, sampled at {sampling_rate} Hz,"
    if method_cleaning in [
        "elgendi",
        "elgendi2013",
    ]:
        report_info["text_cleaning"] += (
            " was preprocessed using a bandpass filter ([0.5 - 8 Hz], Butterworth 3rd order;"
            + " following Elgendi et al., 2013)."
        )
        refs.append(
            """Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013)
            Systolic Peak Detection in Acceleration Photoplethysmograms
            Measured from Emergency Responders in Tropical Conditions
            PLoS ONE 8(10): e76585. doi:10.1371/journal.pone.0076585."""
        )
    elif method_cleaning in ["nabian", "nabian2018"]:
        if report_info["heart_rate"] is None:
            cutoff = "of 40 Hz"
        else:
            cutoff = f' based on the heart rate of {report_info["heart_rate"]} bpm'

        report_info["text_cleaning"] = (
            f" was preprocessed using a lowpass filter (with a cutoff frequency {cutoff},"
            + " butterworth 2nd order; following Nabian et al., 2018)."
        )
        refs.append(
            """Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., & Ostadabbas, S.(2018).
            An open-source feature extraction tool for the analysis of peripheral physiological data.
            IEEE Journal of Translational Engineering in Health and Medicine, 6, 1-11."""
        )
    elif method_cleaning in ["none"]:
        report_info[
            "text_cleaning"
        ] += " was directly used for peak detection without preprocessing."
    else:
        # just in case more methods are added
        report_info["text_cleaning"] = (
            "was cleaned following the " + method + " method."
        )

    # 2. Peaks
    # ----------
    if method_peaks in ["elgendi", "elgendi13"]:
        report_info[
            "text_peaks"
        ] = "The peak detection was carried out using the method described in Elgendi et al. (2013)."
        refs.append(
            """Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013)
            Systolic Peak Detection in Acceleration Photoplethysmograms
            Measured from Emergency Responders in Tropical Conditions
            PLoS ONE 8(10): e76585. doi:10.1371/journal.pone.0076585."""
        )
    elif method_peaks in ["none"]:
        report_info["text_peaks"] = "There was no peak detection carried out."
    else:
        report_info[
            "text_peaks"
        ] = f"The peak detection was carried out using the method {method_peaks}."

    # 2. Quality
    # ----------
    if method_quality in ["templatematch"]:
        report_info[
            "text_quality"
        ] = (
            "The quality assessment was carried out using template-matching, approximately as described "
            + "in Orphanidou et al. (2015)."
        )
        refs.append(
            """Orphanidou C, Bonnici T, Charlton P, Clifton D, Vallance D, Tarassenko L (2015)
            Signal-quality indices for the electrocardiogram and photoplethysmogram: Derivation
            and applications to wireless monitoring
            IEEE Journal of Biomedical and Health Informatics 19(3): 832–838. doi:10.1109/JBHI.2014.2338351."""
        )
    elif method_quality in ["disimilarity"]:
        report_info[
            "text_quality"
        ] = (
            "The quality assessment was carried out using a disimilarity measure of positive-peaked beats, "
            + "approximately as described in Sabeti et al. (2019)."
        )
        refs.append(
            """Sabeti E, Reamaroon N, Mathis M, Gryak J, Sjoding M, Najarian K (2019)
            Signal quality measure for pulsatile physiological signals using
            morphological features: Applications in reliability measure for pulse oximetry
            Informatics in Medicine Unlocked 16: 100222. doi:10.1016/j.imu.2019.100222."""
        )
    elif method_quality in ["none"]:
        report_info["text_quality"] = "There was no quality assessment carried out."
    else:
        report_info[
            "text_quality"
        ] = f"The quality assessment was carried out using the method {method_quality}."

    report_info["references"] = list(np.unique(refs))
    return report_info

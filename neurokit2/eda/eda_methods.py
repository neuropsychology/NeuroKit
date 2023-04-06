# -*- coding: utf-8 -*-
import numpy as np

from ..misc.report import get_kwargs
from .eda_clean import eda_clean
from .eda_peaks import eda_peaks
from .eda_phasic import eda_phasic

def eda_methods(
    sampling_rate=1000,
    method="default",
    method_cleaning="default",
    method_peaks="default",
    method_phasic="default",
    **kwargs,
):
    """**EDA Preprocessing Methods**

    This function analyzes and specifies the methods used in the preprocessing, and create a
    textual description of the methods used. It is used by :func:`eda_process()` to dispatch the
    correct methods to each subroutine of the pipeline and :func:`eda_report()` to create a
    preprocessing report.

    Parameters
    ----------
    sampling_rate : int
        The sampling frequency of the raw EDA signal (in Hz, i.e., samples/second).
    method : str
        The method used for cleaning and peak finding if ``"method_cleaning"``
        and ``"method_peaks"`` are set to ``"default"``. Can be one of ``"default"``, ``"biosppy"``.
        Defaults to ``"default"``.
    method_cleaning: str
        The method used to clean the raw EDA signal. If ``"default"``,
        will be set to the value of ``"method"``. Defaults to ``"default"``.
        For more information, see the ``"method"`` argument
        of :func:`.eda_clean`.
    method_peaks: str
        The method used to find peaks. If ``"default"``,
        will be set to the value of ``"method"``. Defaults to ``"default"``.
        For more information, see the ``"method"`` argument
        of :func:`.eda_peaks`.
    method_phasic: str
        The method used to decompose the EDA signal into phasic and tonic components. If ``"default"``,
        will be set to the value of ``"method"``. Defaults to ``"default"``.
        For more information, see the ``"method"`` argument
        of :func:`.eda_phasic`.
    **kwargs
        Other arguments to be passed to :func:`.eda_clean`,
        :func:`.eda_peaks`, and :func:`.eda_phasic`.

    Returns
    -------
    report_info : dict
        A dictionary containing the keyword arguments passed to the cleaning
        and peak finding functions, text describing the methods, and the corresponding
        references.

    See Also
    --------
    eda_process, eda_clean, eda_peaks
    """
    # Sanitize inputs
    method_cleaning = str(method).lower() if method_cleaning == "default" else str(method_cleaning).lower()
    method_phasic = str(method).lower() if method_phasic == "default" else str(method_phasic).lower()
    method_peaks = str(method).lower() if method_peaks == "default" else str(method_peaks).lower()

    # Create dictionary with all inputs
    report_info = {
        "sampling_rate": sampling_rate,
        "method_cleaning": method_cleaning,
        "method_phasic": method_phasic,
        "method_peaks": method_peaks,
        "kwargs": kwargs,
    }

    # Get arguments to be passed to underlying functions
    kwargs_cleaning, report_info = get_kwargs(report_info, eda_clean)
    kwargs_phasic, report_info = get_kwargs(report_info, eda_phasic)
    kwargs_peaks, report_info = get_kwargs(report_info, eda_peaks)

    # Save keyword arguments in dictionary
    report_info["kwargs_cleaning"] = kwargs_cleaning
    report_info["kwargs_phasic"] = kwargs_phasic
    report_info["kwargs_peaks"] = kwargs_peaks

    # Initialize refs list
    refs = []

    # 1. Cleaning
    # ------------
    report_info["text_cleaning"] = f"The raw signal, sampled at {sampling_rate} Hz,"
    if method_cleaning == "biosppy":
        report_info["text_cleaning"] += " was cleaned using the biosppy package."
    elif method_cleaning in ["default", "neurokit", "nk"]:
        report_info["text_cleaning"] += " was cleaned using the default method of the neurokit2 package."
    elif method_cleaning in ["none"]:
        report_info["text_cleaning"] += "was directly used without cleaning."
    else:
        report_info["text_cleaning"] += " was cleaned using the method described in " + method_cleaning + "."

    # 2. Phasic decomposition
    # -----------------------
    # TODO: add descriptions of individual methods
    report_info["text_phasic"] = "The signal was decomposed into phasic and tonic components using"
    if method_phasic is None or method_phasic in ["none"]:
        report_info["text_phasic"] = "There was no phasic decomposition carried out."
    else:
        report_info["text_phasic"] += " the method described in " + method_phasic + "."

    # 3. Peak detection
    # -----------------
    report_info["text_peaks"] = "The cleaned signal was used to detect peaks using"
    if method_peaks in ["gamboa2008", "gamboa"]:
        report_info["text_peaks"] += " the method described in Gamboa et al. (2008)."
        refs.append("""Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci
        and electrophysiology. PhD ThesisUniversidade.""")
    elif method_peaks in ["kim", "kbk", "kim2004", "biosppy"]:
        report_info["text_peaks"] += " the method described in Kim et al. (2004)."
        refs.append("""Kim, K. H., Bang, S. W., & Kim, S. R. (2004). Emotion recognition system using short-term
      monitoring of physiological signals. Medical and biological engineering and computing, 42(3),
      419-427.""")
    elif method_peaks in ["nk", "nk2", "neurokit", "neurokit2"]:
        report_info["text_peaks"] += " the default method of the `neurokit2` package."
        refs.append("https://doi.org/10.21105/joss.01667")
    elif method_peaks in ["vanhalem2020", "vanhalem", "halem2020"]:
        report_info["text_peaks"] += " the method described in Vanhalem et al. (2020)."
        refs.append("""van Halem, S., Van Roekel, E., Kroencke, L., Kuper, N., & Denissen, J. (2020).
      Moments That Matter? On the Complexity of Using Triggers Based on Skin Conductance to Sample
      Arousing Events Within an Experience Sampling Framework. European Journal of Personality.""")
    elif method_peaks in ["nabian2018", "nabian"]:
        report_info["text_peaks"] += " the method described in Nabian et al. (2018)."
        refs.append("""Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., & Ostadabbas, S. (2018). An
      Open-Source Feature Extraction Tool for the Analysis of Peripheral Physiological Data. IEEE
      journal of translational engineering in health and medicine, 6, 2800711.""")
    else:
        report_info[
            "text_peaks"
        ] = f"The peak detection was carried out using the method {method_peaks}."

    # References
    report_info["references"] = list(np.unique(refs))
    return report_info

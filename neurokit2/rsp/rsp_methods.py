# -*- coding: utf-8 -*-
import numpy as np

from ..misc.report import get_kwargs
from .rsp_clean import rsp_clean
from .rsp_peaks import rsp_peaks
from .rsp_rvt import rsp_rvt


def rsp_methods(
    sampling_rate=1000,
    method="khodadad",
    method_cleaning="default",
    method_peaks="default",
    method_rvt="power",
    **kwargs,
):
    """**RSP Preprocessing Methods**

    This function analyzes and specifies the methods used in the preprocessing, and create a
    textual description of the methods used. It is used by :func:`rsp_process()` to dispatch the
    correct methods to each subroutine of the pipeline and :func:`rsp_report()` to create a
    preprocessing report.

    Parameters
    ----------
    sampling_rate : int
        The sampling frequency of the raw RSP signal (in Hz, i.e., samples/second).
    method : str
        The method used for cleaning and peak finding if ``"method_cleaning"``
        and ``"method_peaks"`` are set to ``"default"``. Can be one of ``"Khodadad"``, ``"BioSPPy"``.
        Defaults to ``"Khodadad"``.
    method_cleaning: str
        The method used to clean the raw RSP signal. If ``"default"``,
        will be set to the value of ``"method"``. Defaults to ``"default"``.
        For more information, see the ``"method"`` argument
        of :func:`.rsp_clean`.
    method_peaks: str
        The method used to find peaks. If ``"default"``,
        will be set to the value of ``"method"``. Defaults to ``"default"``.
        For more information, see the ``"method"`` argument
        of :func:`.rsp_peaks`.
    method_rvt: str
        The method used to compute respiratory volume per time. Defaults to ``"harrison"``.
        For more information, see the ``"method"`` argument
        of :func:`.rsp_rvt`.
    **kwargs
        Other arguments to be passed to :func:`.rsp_clean`,
        :func:`.rsp_peaks`, and :func:`.rsp_rvt`.

    Returns
    -------
    report_info : dict
        A dictionary containing the keyword arguments passed to the cleaning
        and peak finding functions, text describing the methods, and the corresponding
        references.

    See Also
    --------
    rsp_process, rsp_clean, rsp_findpeaks

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      methods = nk.rsp_methods(sampling_rate=100, method="Khodadad", method_cleaning="hampel")
      print(methods["text_cleaning"])
      print(methods["references"][0])
    """
    # Sanitize inputs
    method_cleaning = (
        str(method).lower() if method_cleaning == "default" else str(method_cleaning).lower()
    )
    method_peaks = str(method).lower() if method_peaks == "default" else str(method_peaks).lower()
    method_rvt = str(method_rvt).lower()

    # Create dictionary with all inputs
    report_info = {
        "sampling_rate": sampling_rate,
        "method": method,
        "method_cleaning": method_cleaning,
        "method_peaks": method_peaks,
        "method_rvt": method_rvt,
        **kwargs,
    }

    # Get arguments to be passed to cleaning and peak finding functions
    kwargs_cleaning, report_info = get_kwargs(report_info, rsp_clean)
    kwargs_peaks, report_info = get_kwargs(report_info, rsp_peaks)
    kwargs_rvt, report_info = get_kwargs(report_info, rsp_rvt)

    # Save keyword arguments in dictionary
    report_info["kwargs_cleaning"] = kwargs_cleaning
    report_info["kwargs_peaks"] = kwargs_peaks
    report_info["kwargs_rvt"] = kwargs_rvt

    # Initialize refs list with NeuroKit2 reference
    refs = ["""Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H.,
    Schölzel, C., & Chen, S. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing.
    Behavior Research Methods, 53(4), 1689–1696. https://doi.org/10.3758/s13428-020-01516-y
    """]

    # 1. Cleaning
    # ------------
    report_info["text_cleaning"] = f"The raw signal, sampled at {sampling_rate} Hz,"
    if method_cleaning in ["khodadad", "khodadad2018"]:
        report_info["text_cleaning"] += (
            " was preprocessed using a second order 0.05-3 Hz bandpass Butterworth filter."
        )
    elif method_cleaning in ["hampel", "power", "power2020"]:
        report_info["text_cleaning"] += (
            " was preprocessed using a median-based Hampel filter by replacing values which"
            + f' are {report_info.get("threshold", 3)} median absolute deviation away from the rolling median;'
            + "following Power et al. 2020."
        )

        refs.append(
            """Power, J., Lynch, C., Dubin, M., Silver, B., Martin, A., Jones, R.,(2020)
            Characteristics of respiratory measures in young adults scanned at rest,
            including systematic changes and “missed” deep breaths.
            NeuroImage, Volume 204, 116234"""
        )
    elif method_cleaning in ["biosppy"]:
        report_info["text_cleaning"] += (
            " was preprocessed using a second order 0.1-0.35 Hz bandpass "
            + "Butterworth filter followed by a constant detrending."
        )
    elif method_cleaning in ["none"]:
        report_info[
            "text_cleaning"
        ] += " was directly used for peak detection without preprocessing."
    else:
        # just in case more methods are added
        report_info["text_cleaning"] += f" was cleaned following the {method} method."

    # 2. Peaks
    # ----------
    if method_peaks in ["khodadad", "khodadad2018"]:
        report_info[
            "text_peaks"
        ] = "The peak detection was carried out using the method described in Khoadadad et al. (2018)."
        refs.append(
            """Khodadad, D., Nordebo, S., Müller, B., Waldmann, A., Yerworth, R., Becher, T., ... & Bayford, R. (2018).
            Optimized breath detection algorithm in electrical impedance tomography.
            Physiological measurement, 39(9), 094001."""
        )
    elif method_peaks in ["biosppy"]:
        report_info[
            "text_peaks"
        ] = "The peak detection was carried out using the method provided by the Python library BioSPpy (/signals/resp.py)."
    elif method_peaks in ["scipy"]:
        report_info[
            "text_peaks"
        ] = "The peak detection was carried out using the method provided by the Python library SciPy (signal.find_peaks)."
    elif method_peaks in ["none"]:
        report_info["text_peaks"] = "There was no peak detection carried out."
    else:
        report_info[
            "text_peaks"
        ] = f"The peak detection was carried out using the method {method_peaks}."

    # 3. RVT
    # ----------
    if method_rvt in ["harrison", "harrison2021"]:
        report_info[
            "text_rvt"
        ] = "The respiratory volume per time computation was carried out using the method described in Harrison et al. (2021)."
        refs.append(
            """Harrison, S. J., Bianchi, S., Heinzle, J., Stephan, K. E., Iglesias, S., & Kasper, L. (2021).
            A Hilbert-based method for processing respiratory timeseries.
            Neuroimage, 230, 117787."""
        )
    elif method_rvt in ["birn", "birn2006"]:
        report_info[
            "text_rvt"
        ] = "The respiratory volume per time computation was carried out using the method described in Birn et al. (2006)."
        refs.append(
            """Birn, R. M., Diamond, J. B., Smith, M. A., & Bandettini, P. A. (2006).
            Separating respiratory-variation-related fluctuations from neuronal-activity-related fluctuations in
            fMRI. Neuroimage, 31(4), 1536-1548."""
        )
    elif method_rvt in ["power", "power2020"]:
        report_info[
            "text_rvt"
        ] = "The respiratory volume per time computation was carried out using the method described in Power at al. (2020)."
        refs.append(
            """Power, J. D., Lynch, C. J., Dubin, M. J., Silver, B. M., Martin, A., & Jones, R. M. (2020).
            Characteristics of respiratory measures in young adults scanned at rest, including systematic
            changes and "missed" deep breaths. Neuroimage, 204, 116234."""
        )
    elif method_rvt in ["none"]:
        report_info["text_rvt"] = "Respiratory volume per time was not computed."
    else:
        report_info[
            "text_rvt"
        ] = f"The respiratory volume per time computation was carried out using the method described in {method_rvt}."

    report_info["references"] = list(np.unique(refs))
    return report_info

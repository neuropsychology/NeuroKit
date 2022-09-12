# -*- coding: utf-8 -*-
import numpy as np

from ..report import get_default_args
from .ppg_clean import ppg_clean
from .ppg_findpeaks import ppg_findpeaks


def ppg_report(
    sampling_rate=1000,
    method="elgendi",
    method_cleaning="default",
    method_peaks="default",
    **kwargs
):
    """**Sanitize and describe methods for processing a PPG signal.**

    This function first sanitizes the input, i.e., 
    if the specific methods are "default"
    then it adjusts based on the "general" default
    And then it creates the pieces of text for each method.

    Parameters
    ----------
    sampling_rate : int
        The sampling frequency of the raw PPG signal (in Hz, i.e., samples/second).
    method : str
        The method used for cleaning and peak detection if ``"method_cleaning"``
        and ``"method_peaks"`` are set to ``"default"``. Can be one of ``"elgendi"``. 
        Defaults to ``"elgendi"``.
    """

    if method_cleaning == "default":
        method_cleaning = method
    if method_peaks == "default":
        method_peaks = method
    defaults_cleaning = get_default_args(ppg_clean)
    defaults_peaks = get_default_args(ppg_findpeaks)

    report_info = {
        "sampling_rate": sampling_rate,
        "method": method,
        "method_cleaning": method_cleaning,
        "method_peaks": method_peaks,
        **kwargs,
    }

    kwargs_cleaning = {}
    for key in defaults_cleaning.keys():
        # if arguments have not already been specified
        if key not in ["sampling_rate", "method"]:
            if key not in report_info.keys():
                report_info[key] = defaults_cleaning[key]
            elif report_info[key] != defaults_cleaning[key]:
                kwargs_cleaning[key] = report_info[key]
    kwargs_peaks = {}
    for key in defaults_peaks.keys():
        # if arguments have not already been specified
        if key not in ["sampling_rate", "method"]:
            if key not in report_info.keys():
                report_info[key] = defaults_peaks[key]
            elif report_info[key] != defaults_peaks[key]:
                kwargs_peaks[key] = report_info[key]
    # could also specify parameters if they are not defaults
    report_info["kwargs_cleaning"] = kwargs_cleaning
    report_info["kwargs_peaks"] = kwargs_peaks

    refs = []

    if method_cleaning in ["elgendi"]:
        report_info["text_cleaning"] = (
            """The data cleaning was performed using the Elgendi et al. (2013) method: 
                         the raw PPG signal (sampled at """
            + str(report_info["sampling_rate"])
            + """ Hz) was filtered with a bandpass filter ([0.5, 8], butterworth 3rd order).
                         """
        )
        refs.append(
            """Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection in
          Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions.
          PLoS ONE 8(10): e76585. doi:10.1371/journal.pone.0076585."""
        )
    elif method_cleaning in ["nabian2018"]:
        if report_info["heart_rate"] is None:
            text_cleaning_cutoff = "cutoff = 40"
        else:
            text_cleaning_cutoff = (
                "cutoff frequency determined based on provided heart rate of "
                + str(report_info["heart_rate"])
            )
        report_info["text_cleaning"] = (
            """The data cleaning was performed using the Nabian et al. (2018) method: 
                         the raw PPG signal (sampled at """
            + str(report_info["sampling_rate"])
            + """ Hz) was filtered with a lowpass filter ("""
            + text_cleaning_cutoff
            + """, butterworth 2nd order).
                         """
        )
        refs.append(
            """Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., &amp; Ostadabbas, S.
          (2018). An Open-Source Feature Extraction Tool for the Analysis of Peripheral Physiological
          Data. IEEE Journal of Translational Engineering in Health and Medicine, 6, 1-11. doi:10.1109/jtehm.2018.2878000"""
        )
    else:
        # just in case more methods are added
        report_info["text_cleaning"] = (
            "The data cleaning was performed using the " + method + " method."
        )
    if method_peaks in ["elgendi"]:
        report_info[
            "text_peaks"
        ] = "The peak detection was carried out using the Elgendi et al. (2013) method."
        refs.append(
            """Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection in
          Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions.
          PLoS ONE 8(10): e76585. doi:10.1371/journal.pone.0076585."""
        )
    report_info["references"] = list(np.unique(refs))

    return report_info

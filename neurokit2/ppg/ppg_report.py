# -*- coding: utf-8 -*-


def ppg_report(method="elgendi", method_cleaning="default", method_peaks="default"):
    # This function first sanitizes the input, i.e., if the specific methods are "default" then it adjusts based on the "general" default
    # And then it creates the pieces of text for each element
    text_cleaning = ""
    text_peaks = ""
    return {
        "method_cleaning": method_cleaning,
        "method_peaks": method_peaks,
        "text_cleaning": text_cleaning,
        "text_peaks": text_peaks,
    }

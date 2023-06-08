# -*- coding: utf-8 -*-
import numpy as np

from ..misc.report import get_kwargs
from .emg_activation import emg_activation
from .emg_clean import emg_clean


def emg_methods(
    sampling_rate=1000,
    method_cleaning="biosppy",
    method_activation="threshold",
    **kwargs,
):
    """**EMG Preprocessing Methods**

    This function analyzes and specifies the methods used in the preprocessing, and create a
    textual description of the methods used. It is used by :func:`eda_process()` to dispatch the
    correct methods to each subroutine of the pipeline and to create a
    preprocessing report.

    Parameters
    ----------
    sampling_rate : int
        The sampling frequency of the raw EMG signal (in Hz, i.e., samples/second).
    method_cleaning : str
        The method used for cleaning the raw EMG signal. Can be one of ``"biosppy"`` or ``"none"``.
        Defaults to ``"biosppy"``. If ``"none"`` is passed, the raw signal will be used without
        any cleaning.
    method_activation: str
        The method used for locating EMG activity. Defaults to ``"threshold"``.
        For more information, see the ``"method"`` argument
        of :func:`.emg_activation`.
    **kwargs
        Other arguments to be passed to :func:`.emg_activation`,

    Returns
    -------
    report_info : dict
        A dictionary containing the keyword arguments passed to the cleaning and activation
        functions, text describing the methods, and the corresponding references.

    See Also
    --------

    """
    # Sanitize inputs
    method_cleaning = str(method_cleaning).lower()
    method_activation = str(method_activation).lower()

    # Create dictionary with all inputs
    report_info = {
        "sampling_rate": sampling_rate,
        "method_cleaning": method_cleaning,
        "method_activation": method_activation,
        **kwargs,
    }

    # Get arguments to be passed to activation function
    kwargs_activation, report_info = get_kwargs(report_info, emg_activation)

    # Save keyword arguments in dictionary
    report_info["kwargs_activation"] = kwargs_activation

    # Initialize refs list
    refs = []

    # 1. Cleaning
    # ------------
    # If no cleaning
    report_info["text_cleaning"] = f"The raw signal, sampled at {sampling_rate} Hz,"
    if method_cleaning in ["none"]:
        report_info["text_cleaning"] += " was directly used without any cleaning."
    else:
        report_info["text_cleaning"] += (
            " was cleaned using the " + method_cleaning + " method."
        )

    # 2. Activation
    # -------------
    report_info["text_activation"] = (
        f"EMG activity was detected using the " + method_activation + " method."
    )

    # 3. References
    # -------------
    report_info["refs"] = list(np.unique(refs))

    return report_info

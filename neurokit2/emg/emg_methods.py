# -*- coding: utf-8 -*-
import numpy as np

from ..misc.report import get_kwargs
from .emg_activation import emg_activation


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

    # Initialize refs list with NeuroKit2 reference
    refs = [
        """Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H.,
    Schölzel, C., & Chen, S. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing.
    Behavior Research Methods, 53(4), 1689–1696. https://doi.org/10.3758/s13428-020-01516-y
    """
    ]

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
        "EMG activity was detected using the " + method_activation + " method. "
    )
    if method_activation in ["silva"]:
        if str(report_info["threshold"]) == "default":
            threshold_str = "0.05"
        else:
            threshold_str = str(report_info["threshold"])
        report_info["text_activation"] += f"""The threshold was {threshold_str}. """

        refs.append(
            """Silva H, Scherer R, Sousa J, Londral A , "Towards improving the ssability of
            electromyographic interfacess", Journal of Oral Rehabilitation, pp. 1-2, 2012."""
        )
    if method_activation in ["mixture"]:
        report_info[
            "text_activation"
        ] += """A Gaussian mixture model was used to discriminate between activity and baseline. """
        if str(report_info["threshold"]) == "default":
            threshold_str = "0.33"
        else:
            threshold_str = str(report_info["threshold"])
        report_info[
            "text_activation"
        ] += f"""The minimum probability required to
        be considered as activated was {threshold_str}. """
    elif method_activation in ["threshold"]:
        report_info[
            "text_activation"
        ] += """The signal was considered as activated when the amplitude exceeded a threshold. """
        if str(report_info["threshold"]) == "default":
            threshold_str = "one tenth of the standard deviation of emg_amplitude"
        else:
            threshold_str = str(report_info["threshold"])
        report_info[
            "text_activation"
        ] += f"""The minimum amplitude to detect as onset was set to {threshold_str}."""
    elif method_activation in ["biosppy"]:
        if str(report_info["threshold"]) == "default":
            threshold_str = "1.2 times of the mean of the absolute of the smoothed, full-wave-rectified signal"
        else:
            threshold_str = str(report_info["threshold"])
        report_info[
            "text_activation"
        ] += f"""The threshold was set to {threshold_str}."""

    # 3. References
    # -------------
    report_info["references"] = list(np.unique(refs))

    return report_info

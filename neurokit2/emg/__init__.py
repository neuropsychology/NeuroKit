"""Submodule for NeuroKit."""

from .emg_activation import emg_activation
from .emg_amplitude import emg_amplitude
from .emg_analyze import emg_analyze
from .emg_clean import emg_clean
from .emg_eventrelated import emg_eventrelated
from .emg_intervalrelated import emg_intervalrelated
from .emg_plot import emg_plot
from .emg_process import emg_process
from .emg_simulate import emg_simulate


__all__ = [
    "emg_simulate",
    "emg_clean",
    "emg_amplitude",
    "emg_process",
    "emg_plot",
    "emg_activation",
    "emg_eventrelated",
    "emg_intervalrelated",
    "emg_analyze",
]

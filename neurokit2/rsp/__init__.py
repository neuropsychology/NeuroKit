"""
Submodule for NeuroKit.
"""

# Aliases
from ..signal import signal_rate as rsp_rate
from .rsp_amplitude import rsp_amplitude
from .rsp_analyze import rsp_analyze
from .rsp_clean import rsp_clean
from .rsp_eventrelated import rsp_eventrelated
from .rsp_findpeaks import rsp_findpeaks
from .rsp_fixpeaks import rsp_fixpeaks
from .rsp_intervalrelated import rsp_intervalrelated
from .rsp_peaks import rsp_peaks
from .rsp_phase import rsp_phase
from .rsp_plot import rsp_plot
from .rsp_process import rsp_process
from .rsp_rrv import rsp_rrv
from .rsp_simulate import rsp_simulate


__all__ = [
    "rsp_simulate",
    "rsp_clean",
    "rsp_findpeaks",
    "rsp_fixpeaks",
    "rsp_peaks",
    "rsp_phase",
    "rsp_amplitude",
    "rsp_process",
    "rsp_plot",
    "rsp_eventrelated",
    "rsp_rrv",
    "rsp_intervalrelated",
    "rsp_analyze",
    "rsp_rate",
]

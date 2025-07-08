"""Submodule for NeuroKit."""

from .signal_autocor import signal_autocor
from .signal_binarize import signal_binarize
from .signal_changepoints import signal_changepoints
from .signal_cyclesegment import signal_cyclesegment
from .signal_decompose import signal_decompose
from .signal_detrend import signal_detrend
from .signal_distort import signal_distort
from .signal_fillmissing import signal_fillmissing
from .signal_filter import signal_filter
from .signal_findpeaks import signal_findpeaks
from .signal_fixpeaks import signal_fixpeaks
from .signal_flatline import signal_flatline
from .signal_formatpeaks import signal_formatpeaks
from .signal_interpolate import signal_interpolate
from .signal_merge import signal_merge
from .signal_noise import signal_noise
from .signal_period import signal_period
from .signal_phase import signal_phase
from .signal_plot import signal_plot
from .signal_power import signal_power
from .signal_psd import signal_psd
from .signal_rate import signal_rate
from .signal_recompose import signal_recompose
from .signal_resample import signal_resample
from .signal_sanitize import signal_sanitize
from .signal_simulate import signal_simulate
from .signal_smooth import signal_smooth
from .signal_surrogate import signal_surrogate
from .signal_synchrony import signal_synchrony
from .signal_quality import signal_quality
from .signal_tidypeaksonsets import signal_tidypeaksonsets
from .signal_timefrequency import signal_timefrequency
from .signal_zerocrossings import signal_zerocrossings

__all__ = [
    "signal_simulate",
    "signal_binarize",
    "signal_cyclesegment",
    "signal_resample",
    "signal_zerocrossings",
    "signal_smooth",
    "signal_filter",
    "signal_psd",
    "signal_distort",
    "signal_interpolate",
    "signal_detrend",
    "signal_findpeaks",
    "signal_fixpeaks",
    "signal_formatpeaks",
    "signal_rate",
    "signal_merge",
    "signal_noise",
    "signal_period",
    "signal_plot",
    "signal_phase",
    "signal_power",
    "signal_synchrony",
    "signal_autocor",
    "signal_changepoints",
    "signal_decompose",
    "signal_recompose",
    "signal_surrogate",
    "signal_tidypeaksonsets",
    "signal_quality",
    "signal_timefrequency",
    "signal_sanitize",
    "signal_flatline",
    "signal_fillmissing",
]

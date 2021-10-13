"""Submodule for NeuroKit."""

from ._warnings import NeuroKitWarning
from .check_type import check_type
from .expspace import expspace
from .find_closest import find_closest
from .find_consecutive import find_consecutive
from .find_groups import find_groups
from .find_outliers import find_outliers
from .find_plateau import find_plateau
from .intervals_to_peaks import intervals_to_peaks
from .listify import listify
from .parallel_run import parallel_run
from .replace import replace
from .type_converters import as_vector

__all__ = [
    "listify",
    "find_closest",
    "find_consecutive",
    "find_groups",
    "as_vector",
    "expspace",
    "replace",
    "NeuroKitWarning",
    "check_type",
    "find_outliers",
    "intervals_to_peaks",
    "parallel_run",
    "find_plateau",
]

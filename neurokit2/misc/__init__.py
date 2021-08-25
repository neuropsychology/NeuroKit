"""Submodule for NeuroKit."""

from .expspace import expspace
from .find_closest import find_closest
from .find_consecutive import find_consecutive
from .find_groups import find_groups
from .listify import listify
from .type_converters import as_vector
from .replace import replace
from .warnings import NeuroKitWarning
from .check_type import check_type
from .find_outliers import find_outliers
from .intervals_to_peaks import intervals_to_peaks


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
    "intervals_to_peaks"
]

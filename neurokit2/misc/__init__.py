"""Submodule for NeuroKit."""

from .expspace import expspace
from .find_closest import find_closest
from .find_consecutive import find_consecutive
from .find_groups import find_groups
from .listify import listify
from .type_converters import as_vector
from .replace import replace
from .warnings import NeuroKitWarning


__all__ = [
    "listify",
    "find_closest",
    "find_consecutive",
    "find_groups",
    "as_vector",
    "expspace",
    "replace",
    "NeuroKitWarning",
]

"""Top-level package for NeuroKit."""

# Info
__version__ = '0.0.9'
__citation__ = 'Not yet available.'
__bibtex__ = __citation__
__cite__ = __citation__


# Maintainer info
__author__ = """NeuroKit development team"""
__email__ = 'dom.makowski@gmail.com'



# Export content of submodules
from .misc import *
from .stats import *
from .complexity import *
from .signal import *
from .events import *
from .epochs import *
from .data import *

from .ecg import *
from .rsp import *
from .eda import *
from .emg import *
from .ppg import *
from .eeg import *
from .bio import *

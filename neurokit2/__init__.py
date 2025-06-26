"""Top-level package for NeuroKit."""

import datetime
import platform

import matplotlib

# Dependencies
import numpy as np
import pandas as pd
import scipy
import sklearn

from .benchmark import *
from .bio import *
from .complexity import *
from .data import *
from .ecg import *
from .eda import *
from .eeg import *
from .emg import *
from .eog import *
from .epochs import *
from .events import *
from .hrv import *
from .markov import *
from .microstates import *
from .misc import *
from .ppg import *
from .rsp import *
from .signal import *
from .stats import *
from .video import *

# Info
__version__ = "0.2.12"


# Maintainer info
__author__ = "The NeuroKit development team"
__email__ = "D.Makowski@sussex.ac.uk"


# Citation
__bibtex__ = r"""
@article{Makowski2021neurokit,
    author = {Dominique Makowski and Tam Pham and Zen J. Lau and Jan C. Brammer and Fran{\c{c}}ois Lespinasse and Hung Pham and Christopher Schölzel and S. H. Annabel Chen},
    title = {{NeuroKit}2: A Python toolbox for neurophysiological signal processing},
    journal = {Behavior Research Methods},
    volume = {53},
    number = {4},
    pages = {1689--1696},
    publisher = {Springer Science and Business Media {LLC}},
    doi = {10.3758/s13428-020-01516-y},
    url = {https://doi.org/10.3758%2Fs13428-020-01516-y},
    year = 2021,
    month = {feb}
}
"""

__cite__ = (
    """
You can cite NeuroKit2 as follows:

- Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H.,
Schölzel, C., & Chen, S. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing.
Behavior Research Methods, 53(4), 1689-1696. https://doi.org/10.3758/s13428-020-01516-y


Full bibtex reference:
"""
    + __bibtex__
)
# Aliases for citation
__citation__ = __cite__


# =============================================================================
# Helper functions to retrieve info
# =============================================================================
def cite(silent=False):
    """Cite NeuroKit2.

    This function will print the bibtex and the APA reference for your to copy and cite.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk
      nk.cite()

    """
    if silent is False:
        print(__cite__)
    else:
        return __bibtex__


def version(silent=False):
    """NeuroKit2's version.

    This function is a helper to retrieve the version of the package.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk
      nk.version()

    """
    if silent is False:
        print(
            "- OS: " + platform.system(),
            "(" + platform.architecture()[1] + " " + platform.architecture()[0] + ")",
            "\n- Python: " + platform.python_version(),
            "\n- NeuroKit2: " + __version__,
            "\n\n- NumPy: " + np.__version__,
            "\n- Pandas: " + pd.__version__,
            "\n- SciPy: " + scipy.__version__,
            "\n- sklearn: " + sklearn.__version__,
            "\n- matplotlib: " + matplotlib.__version__,
        )
    else:
        return __version__

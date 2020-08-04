"""Top-level package for NeuroKit."""
import datetime
import platform

# Dependencies
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib

# Export functions
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
from .misc import *
from .ppg import *
from .rsp import *
from .signal import *
from .stats import *
from .microstates import *


# Info
__version__ = "0.0.40"


# Maintainer info
__author__ = "The NeuroKit development team"
__email__ = "dom.makowski@gmail.com"


# Citation
__bibtex__ = r"""
@misc{neurokit2,
  doi = {10.5281/ZENODO.3597887},
  url = {https://github.com/neuropsychology/NeuroKit},
  author = {Makowski, Dominique and Pham, Tam and Lau, Zen J. and Brammer, Jan C. and Lesspinasse,
            Fran\c{c}ois and Pham, Hung and Schölzel, Christopher and S H Chen, Annabel},
  title = {NeuroKit2: A Python Toolbox for Neurophysiological Signal Processing},
  publisher = {Zenodo},
  month={Mar},
  year = {2020},
}
"""

__cite__ = (
    """
You can cite NeuroKit2 as follows:

- Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lesspinasse, F., Pham, H.,
  Schölzel, C., & S H Chen, A. (2020). NeuroKit2: A Python Toolbox for Neurophysiological
  Signal Processing. Retrieved """
    + datetime.date.today().strftime("%B %d, %Y")
    + """, from https://github.com/neuropsychology/NeuroKit


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
    >>> import neurokit2 as nk
    >>>
    >>> nk.cite()

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
    >>> import neurokit2 as nk
    >>>
    >>> nk.version()

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

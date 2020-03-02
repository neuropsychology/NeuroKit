"""Top-level package for NeuroKit."""
import datetime

# Info
__version__ = '0.0.12'



# Citation
__cite__ = """
You can cite NeuroKit as follows:

- Makowski, D., Pham, T., L Juen, Z., Brammer, J. C., Pham, H., Lesspinasse, F., & S H Chen, A. (2020). NeuroKit: The Python Toolbox for Neurophysiological Signal Processing. Retrieved """ + datetime.date.today().strftime("%B %d, %Y") + """, from https://github.com/neuropsychology/NeuroKit


Full bibtex reference:

@misc{neurokit,
  doi = {10.5281/ZENODO.3597887},
  url = {https://github.com/neuropsychology/NeuroKit},
  author = {Makowski, Dominique and Pham, Tam and L Juen, Zen and Brammer, Jan C. and Pham, Hung and Lesspinasse, Fran\c{c}ois and S H Chen, Annabel},
  title = {NeuroKit: The Python Toolbox for Neurophysiological Signal Processing},
  publisher = {Zenodo},
  year = {2020},
}
"""
__citation__ = __cite__
__bibtex__ = __citation__


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

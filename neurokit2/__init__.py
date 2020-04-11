"""Top-level package for NeuroKit."""
import datetime

# Info
__version__ = '0.0.26'


# Maintainer info
__author__ = 'The NeuroKit development team'
__email__ = 'dom.makowski@gmail.com'




# Citation
__cite__ = """
You can cite NeuroKit2 as follows:

- Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Pham, H., Lesspinasse, F.,
  Schölzel, C., & S H Chen, A. (2020). NeuroKit2: A Python Toolbox for Neurophysiological
  Signal Processing. Retrieved """ + datetime.date.today().strftime("%B %d, %Y") + """, from https://github.com/neuropsychology/NeuroKit


Full bibtex reference:

@misc{neurokit2,
  doi = {10.5281/ZENODO.3597887},
  url = {https://github.com/neuropsychology/NeuroKit},
  author = {Makowski, Dominique and Pham, Tam and Lau, Zen J. and Brammer, Jan C. and Pham, Hung and Lesspinasse, Fran\c{c}ois and Schölzel, Christopher and S H Chen, Annabel},
  title = {NeuroKit2: A Python Toolbox for Neurophysiological Signal Processing},
  publisher = {Zenodo},
  month={Mar},
  year = {2020},
}
"""
__citation__ = __cite__
__bibtex__ = __citation__

def cite():
    print(__cite__)





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

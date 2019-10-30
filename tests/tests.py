import unittest
import doctest
import numpy as np
import pandas as pd
import neurokit as nk

from test_ecg import TestEcg
from test_signal import TestComplexity
from test_miscellaneous import TestMiscellaneous
from test_statistics import TestStatistics

if __name__ == '__main__':
    unittest.main()
    doctest.testmod()




# -*- coding: utf-8 -*-
import time as builtin_time
import pandas as pd
import numpy as np

import platform
import os


def find_creation_date(path):
    """
    Try to get the date that a file was created, falling back to when it was last modified if that's not possible.

    Parameters
    ----------
    path : str
       File's path.

    Returns
    ----------
    creation_date : str
        Time of file creation.

    Example
    ----------
    >>> import neurokit as nk
    >>> import datetime
    >>>
    >>> creation_date = nk.find_creation_date(file)
    >>> creation_date = datetime.datetime.fromtimestamp(creation_date)
    """
    if platform.system() == 'Windows':
        return(os.path.getctime(path))
    else:
        stat = os.stat(path)
        try:
            return(stat.st_birthtime)
        except AttributeError:
            print("Neuropsydia error: get_creation_date(): We're probably on Linux. No easy way to get creation dates here, so we'll settle for when its content was last modified.")
            return(stat.st_mtime)
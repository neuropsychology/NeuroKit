# -*- coding: utf-8 -*-
"""Script for formatting the Lobachevsky University Electrocardiography Database

Steps:
    1. Run this script.
Credits:
    https://github.com/berndporr/py-ecg-detectors/blob/master/tester_MITDB.py by Bernd Porr
"""
import pandas as pd
import numpy as np
import wfdb
import os

#wfdb.rdrecord(pn_dir='https://physionet.org/files/ludb/1.0.0/')
#wfdb.dl_database("ludb", dl_dir=".")
#dbs = wfdb.get_record_list("ludb", records='all')
#wfdb.dl_files("ludb", ".", dbs)


# COULD'NT MANAGE TO DOWNLOAD IT ENTIRELY.
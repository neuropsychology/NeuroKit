#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import re

from setuptools import find_packages, setup


# Utilities
with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("NEWS.rst") as history_file:
    history = history_file.read()
history = history.replace("\n-------------------", "\n^^^^^^^^^^^^^^^^^^^").replace("\n=====", "\n-----")


def find_version():
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format("__version__"),
        open("neurokit2/__init__.py").read(),
    )
    return result.group(1)


# Dependencies
requirements = ["requests", "numpy", "pandas", "scipy", "scikit-learn>=1.0.0", "matplotlib>=3.5.0", "PyWavelets>=1.4.0"]

# Optional Dependencies (only needed / downloaded for testing purposes, for instance to test against some other packages)
setup_requirements = ["pytest-runner", "numpy"]
test_requirements = requirements + [
    "pytest",
    "coverage",
    "bioread",
    "mne",
    "pyentrp",
    "antropy",
    "EntropyHub",
    "nolds",
    "biosppy==0.6.1",
    "cvxopt",
    "PyWavelets>=1.4.0",
    "EMD-signal",
    "numba>=0.61.0",
    "llvmlite>=0.44.0",
    "astropy",
    "plotly",
    "ts2vg",
    "wfdb",
]

# Setup
setup(
    # Info
    name="neurokit2",
    keywords="NeuroKit2, physiology, bodily signals, Python, ECG, EDA, EMG, PPG",
    url="https://github.com/neuropsychology/NeuroKit",
    version=find_version(),
    description="The Python Toolbox for Neurophysiological Signal Processing.",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/x-rst",
    license="MIT license",
    # The name and contact of a maintainer
    author="Dominique Makowski",
    author_email="D.Makowski@sussex.ac.uk",
    # Dependencies
    install_requires=requirements,
    setup_requires=setup_requirements,
    extras_require={"test": test_requirements},
    test_suite="pytest",
    tests_require=test_requirements,
    # Misc
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13"
    ],
)

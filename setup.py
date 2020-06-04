#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import re

from setuptools import find_packages, setup

# Utilities
with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('NEWS.rst') as history_file:
    history = history_file.read()
history = history.replace("\n-------------------", "\n^^^^^^^^^^^^^^^^^^^").replace("\n=====", "\n-----")


def find_version():
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format("__version__"), open('neurokit2/__init__.py').read())
    return result.group(1)


# Dependencies
requirements = ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib']
setup_requirements = ['pytest-runner', 'numpy']
test_requirements = requirements + ['pytest', 'coverage', 'bioread', 'mne', 'pyentrp', 'nolds', 'biosppy', 'cvxopt', 'PyWavelets']

# Setup
setup(
    author="Dominique Makowski",
    author_email='dom.makowski@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="The Python Toolbox for Neurophysiological Signal Processing.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    packages=find_packages(),
    include_package_data=True,
    keywords='neurokit2',
    name='neurokit2',
    setup_requires=setup_requirements,
    test_suite='pytest',
    tests_require=test_requirements,
    url='https://github.com/neuropsychology/NeuroKit',
    version=find_version(),
    zip_safe=False,
)

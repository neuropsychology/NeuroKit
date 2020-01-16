#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import re


# Utilities
with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('NEWS.rst') as history_file:
    history = history_file.read()

def find_version():
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format("__version__"), open('neurokit2/__init__.py').read())
    return result.group(1)



# Dependencies
requirements = ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib']
setup_requirements = ['pytest-runner', 'numpy']
test_requirements = requirements + ['pytest', 'coverage', 'bioread', 'mne', 'pyentrp', 'nolds', 'biosppy', 'cvxopt']

# Setup
setup(
    author="Dominique Makowski",
    author_email='dom.makowski@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
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

.. highlight:: shell

============
Installation
============


Install Python
-----------------

How to get a working, portable distribution of python?

Windows
^^^^^^^^^


1. Download a non-zero version of `Winython <http://winpython.github.io/>`_
2. Install it somewhere (the desktop is a good place). It creates a folder called `WPyXX-xxxx`
3. In the `WPyXX-xxxx` folder, open `WinPython Command Prompt.exe`
4. Run :code:`pip install https://github.com/neuropsychology/NeuroKit/zipball/master`
5. Start `Spyder.exe`


Mac OS
^^^^^^^^^

1. Install `Anaconda <https://www.anaconda.com/download/>`_
2. Open the `terminal <https://www.youtube.com/watch?time_continue=59&v=gk2CgkURkgY>`_
3. Run :code:`source activate root`
4. Run :code:`pip install https://github.com/neuropsychology/NeuroKit/zipball/master`
5. Start `Spyder.exe`



Install NeuroKit
-----------------

If you already have python, you can install NeuroKit by running this command in your terminal:

.. code-block:: console

    pip install neurokit2

This is the preferred method to install NeuroKit, as it will always install the most stable release. It is also possible to install it directly from github:

.. code-block:: console

    pip install https://github.com/neuropsychology/neurokit/zipball/master



If you don't have `pip <https://pip.pypa.io>`_ installed, this `Python installation guide <http://docs.python-guide.org/en/latest/starting/installation/>`_ can guide you through the process.

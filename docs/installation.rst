.. highlight:: shell

Installation
============

.. hint::
   Spotted a typo? Would like to add something or make a correction? Join us by contributing (`see these guides <https://neurokit2.readthedocs.io/en/latest/contributing/index.html>`_).

1. Python
-----------------


Windows
^^^^^^^^^

Winpython
"""""""""

The advantage of Winpython is its portability (i.e., works out of a folder) and default setup (convenient for science).

1. Download a non-zero version of `Winpython <http://winpython.github.io/>`_
2. Install it somewhere (the desktop is a good place). It creates a folder called `WPyXX-xxxx`
3. In the `WPyXX-xxxx` folder, open `WinPython Command Prompt.exe`
4. Run :code:`pip install https://github.com/neuropsychology/NeuroKit/zipball/master`
5. Start `Spyder.exe`

Miniconda or Anaconda
"""""""""""""""""""""

The difference between the two is straightforward, *miniconda* is recommended if you don't have much storage space and you know what you want to install. Similar to Winpython, Anaconda comes with a *base* environment, meaning you have basic packages pre-installed.

1. Download and install `Miniconda or Anaconda <https://www.anaconda.com/download/>`_ (make sure the ``Anaconda3`` directory is similar to this: ``C:\Users\<username>\anaconda3\``)
2. Open the `Anaconda Prompt` (search for it on your computer)
3. Run :code:`conda help` to see your options 

.. Note:: There should be a name in parentheses before your user's directory, e.g. ``(base) C:\Users\<yourusername>``. That is the name of your computing environment. By default, you have a ``base environment``. We don't want that, so create an environment.

4. Run :code:`conda env create <yourenvname>`; activate it every time you open up conda by running :code:`conda activate <yourenvname>`
5. Is pip (package installer for python) installed in this env? Prompt Anaconda using :code:`pip list` it'll show you all the packages installed in that conda env

.. image:: https://raw.github.com/neuropsychology/Neurokit/master/docs/img/tutorial_installation_conda.jpg

Mac OS
^^^^^^^^^

1. Install `Anaconda <https://www.anaconda.com/download/>`_
2. Open the `terminal <https://www.youtube.com/watch?time_continue=59&v=gk2CgkURkgY>`_
3. Run :code:`source activate root`
4. Run :code:`pip install neurokit2`
5. Start `Spyder.exe`



2. NeuroKit
-------------------

If you already have python, you can install NeuroKit by running this command in your terminal:

.. code-block:: console

    pip install neurokit2

This is the preferred method to install NeuroKit, as it will always install the most stable release. It is also possible to install it directly from github:

.. code-block:: console

    pip install https://github.com/neuropsychology/neurokit/zipball/master

.. Hint:: Enjoy living on the edge? You can always install the latest `dev` branch to access some work-in-progress features using ``pip install https://github.com/neuropsychology/neurokit/zipball/dev``


If you don't have `pip <https://pip.pypa.io>`_ installed, this `Python installation guide <http://docs.python-guide.org/en/latest/starting/installation/>`_ can guide you through the process.

.. highlight:: shell

Installation
============

.. hint::
   Spotted a typo? Would like to add something or make a correction? Join us by contributing (`see these guides <https://neurokit2.readthedocs.io/en/latest/contributing/index.html>`_).


Instructions for users without a Python installation
----------------------------------------------------
Since NeuroKit2 is a Python package, let's first make sure that you have a Python installation on
your computing device. Then we can move on to install the NeuroKit2 package itself.


Windows
^^^^^^^^^

Winpython
"""""""""

The advantage of Winpython is its portability (i.e., works out of a folder) and default setup (convenient for science).

1. Download a non-zero version of `Winpython <http://winpython.github.io/>`_
2. Install it somewhere (the desktop is a good place). It creates a folder called `WPyXX-xxxx`
3. In the `WPyXX-xxxx` folder, open `WinPython Command Prompt.exe`
4. Now you can proceed to :ref:`install the NeuroKit2 package<Instructions for users with a Python installation>`

Miniconda or Anaconda
"""""""""""""""""""""

The difference between the two is straightforward, *miniconda* is recommended if you don't have much storage space and you know what you want to install. Similar to Winpython, *Anaconda* comes with a *base* environment, meaning you have basic packages pre-installed.
Here is some `more information <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda>`_ to help you choose between *miniconda* and *Anaconda*.

1. Download and install `Miniconda or Anaconda <https://www.anaconda.com/products/individual>`_ (make sure the ``Anaconda3`` directory is similar to this: ``C:\Users\<username>\anaconda3\``)
2. Open the `Anaconda Prompt` (search for it on your computer)
3. Run :code:`conda help` to see your options

.. Note:: There should be a name in parentheses before your user's directory, e.g. ``(base) C:\Users\<yourusername>``. That is the name of your computing environment. By default, you have a ``base environment``. We don't want that, so create an environment.

4. Run :code:`conda env create <yourenvname>`; activate it every time you open up conda by running :code:`conda activate <yourenvname>`
5. Now you can proceed to :ref:`install the NeuroKit2 package<Instructions for users with a Python installation>`


.. image:: https://raw.github.com/neuropsychology/Neurokit/master/docs/img/tutorial_installation_conda.jpg

Mac OS
^^^^^^^^^

1. Install `Anaconda <https://www.anaconda.com/download/>`_
2. Open the `terminal <https://www.youtube.com/watch?time_continue=59&v=gk2CgkURkgY>`_
3. Run :code:`source activate root`



Instructions for users with a Python installation
--------------------------------------------------

If you have Python installed as part of `Miniconda` or `Anaconda`, please follow these steps:

1. As described in :ref:`above<Miniconda or Anaconda>`, open the `Anaconda Prompt` and activate your conda environment
2. You can now install NeuroKit2 from `conda-forge <https://anaconda.org/conda-forge/neurokit2>`_ by typing

.. code-block:: console

    conda config --add channels conda-forge
    conda install neurokit2


If you use another Python installation, you can simply install NeuroKit2 from `PyPI <https://pypi.org/project/neurokit2/>`_:

.. code-block:: console

    pip install neurokit2

If you don't have `pip <https://pip.pypa.io>`_ installed, this `Python installation guide <http://docs.python-guide.org/en/latest/starting/installation/>`_ can guide you through the process.


`conda` or `pip` are the preferred methods to install NeuroKit2, as they will install the most up-to-date stable release.

It is also possible to install NeuroKit2 directly from GitHub:

.. code-block:: console

    pip install https://github.com/neuropsychology/neurokit/zipball/master

.. Hint:: Enjoy living on the edge? You can always install the latest `dev` branch to access work-in-progress features using ``pip install https://github.com/neuropsychology/neurokit/zipball/dev``
